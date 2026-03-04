import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model

from env import map_dark_states
from .compression import CompressionTransformer


class RAD(nn.Module):
    """
    Recurrent/Compressed Algorithm Distillation with separate s/a/r tokens.

    Tokenization:
    - One state token, one action token, one reward token per transition
    - One final query-state token
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config["device"]
        self.n_transit = config["n_transit"]
        self.context_len = self.n_transit - 1
        self.tokens_per_transition = 3
        self.full_ad_seq_length = self.tokens_per_transition * self.context_len + 1
        self.max_seq_length = config.get("rad_max_seq_length", self.full_ad_seq_length)
        self.forward_context_bucket = max(1, int(config.get("rad_forward_context_bucket", 1)))
        self.mixed_precision = config["mixed_precision"]
        self.grid_size = config["grid_size"]

        if self.max_seq_length <= 1:
            raise ValueError("rad_max_seq_length must be > 1")

        tf_n_embd = config["tf_n_embd"]
        tf_n_head = config.get("tf_n_head", 4)
        tf_n_layer = config.get("tf_n_layer", 4)
        tf_n_inner = config.get("tf_n_inner", config.get("tf_dim_feedforward", tf_n_embd * 4))
        tf_dropout = config.get("tf_dropout", 0.1)
        tf_attn_dropout = config.get("tf_attn_dropout", 0.1)
        tf_attn_impl = config.get("tf_attn_impl", "sdpa")

        # AD transformer
        gpt2_cfg = GPT2Config(
            n_positions=self.max_seq_length,
            n_embd=tf_n_embd,
            n_layer=tf_n_layer,
            n_head=tf_n_head,
            n_inner=tf_n_inner,
            resid_pdrop=tf_dropout,
            embd_pdrop=tf_dropout,
            attn_pdrop=tf_attn_dropout,
            use_cache=False,
        )
        gpt2_cfg._attn_implementation = tf_attn_impl
        self.transformer_model = GPT2Model(gpt2_cfg)
        # We always pass inputs_embeds, so GPT2 token embedding table is unused.
        # Freeze it to avoid DDP unused-parameter errors.
        if hasattr(self.transformer_model, 'wte') and hasattr(self.transformer_model.wte, 'weight'):
            self.transformer_model.wte.weight.requires_grad_(False)

        self.embed_state = nn.Embedding(config["grid_size"] * config["grid_size"], tf_n_embd)
        self.embed_action = nn.Embedding(config["num_actions"], tf_n_embd)
        self.embed_reward = nn.Linear(1, tf_n_embd)
        self.pred_action = nn.Linear(tf_n_embd, config["num_actions"])

        # Compression settings
        self.n_compress_tokens = config.get("n_compress_tokens", 24)
        self.compress_n_layers = config.get("compress_n_layers", 2)
        self.compress_n_heads = config.get("compress_n_heads", 4)
        self.max_gradient_rounds = config.get("max_gradient_rounds", 2)
        self.max_compressions = config.get("max_compressions", None)

        self.compression_transformer = CompressionTransformer(
            d_model=tf_n_embd,
            n_heads=self.compress_n_heads,
            n_layers=self.compress_n_layers,
            n_compress_tokens=self.n_compress_tokens,
            dim_feedforward=tf_n_inner,
            dropout=tf_dropout,
        )
        self._compression_params = tuple(self.compression_transformer.parameters())
        self.latent_type_embedding = nn.Parameter(torch.zeros(1, 1, tf_n_embd))
        nn.init.trunc_normal_(self.latent_type_embedding, std=0.02)

        if self.n_compress_tokens >= self.max_seq_length - 1:
            raise ValueError("n_compress_tokens must be < (rad_max_seq_length - 1)")

        self.loss_fn = nn.CrossEntropyLoss(
            reduction="mean", label_smoothing=config["label_smoothing"]
        )

    def transformer(self, x, return_attentions=False):
        output = self.transformer_model(
            inputs_embeds=x,
            output_attentions=return_attentions,
            return_dict=True,
            use_cache=False,
        )
        attentions = list(output.attentions) if return_attentions else None
        return output.last_hidden_state, attentions

    def _build_context_tokens(self, states, actions, rewards):
        state_ids = map_dark_states(states.to(torch.long), self.grid_size)
        state_tokens = self.embed_state(state_ids)
        action_tokens = self.embed_action(actions.to(torch.long))
        reward_tokens = self.embed_reward(rewards.unsqueeze(-1).to(torch.float))

        batch_size = state_tokens.size(0)
        context_tokens = torch.stack([state_tokens, action_tokens, reward_tokens], dim=2)
        return context_tokens.reshape(batch_size, -1, context_tokens.size(-1))

    def _pack_sar_tokens(self, state_tokens, action_tokens, reward_tokens):
        batch_size = state_tokens.size(0)
        context_tokens = torch.stack([state_tokens, action_tokens, reward_tokens], dim=2)
        return context_tokens.reshape(batch_size, -1, context_tokens.size(-1))

    def _build_query_token(self, query_states):
        query_state_ids = map_dark_states(query_states.to(torch.long), self.grid_size)
        return self.embed_state(query_state_ids).unsqueeze(1)

    def _compress_sequence(self, sequence, compression_round):
        latent = self.compression_transformer(sequence)
        if compression_round >= self.max_gradient_rounds:
            latent = latent.detach()
        return latent

    def _round_up_to_transition_tokens(self, n_tokens):
        if n_tokens <= 0:
            return 0
        tpt = self.tokens_per_transition
        return ((n_tokens + tpt - 1) // tpt) * tpt

    def _round_down_to_transition_tokens(self, n_tokens):
        if n_tokens <= 0:
            return 0
        tpt = self.tokens_per_transition
        return (n_tokens // tpt) * tpt

    def _compression_chunk_tokens(self, token_capacity, available_tokens):
        chunk = self._round_down_to_transition_tokens(token_capacity)
        if chunk <= 0:
            chunk = self.tokens_per_transition
        return min(chunk, available_tokens)

    def _forward_with_compression(self, context_tokens, query_token):
        context_len = context_tokens.shape[1]
        available_without_latent = self.max_seq_length - 1
        available_with_latent = self.max_seq_length - self.n_compress_tokens - 1

        compression_info = {
            "num_compressions": 0,
            "has_latent_prefix": False,
            "visible_context_tokens": context_len,
        }

        if context_len <= available_without_latent:
            full_input = torch.cat([context_tokens, query_token], dim=1)
            out, _ = self.transformer(full_input, return_attentions=False)
            return out, compression_info

        latent_tokens = None
        remaining_context = context_tokens
        compression_round = 0

        while True:
            recent_capacity = available_with_latent if latent_tokens is not None else available_without_latent
            if remaining_context.shape[1] <= recent_capacity:
                break

            if self.max_compressions is not None and compression_round >= self.max_compressions:
                keep_len = recent_capacity
                keep_len = self._round_down_to_transition_tokens(keep_len)
                keep_len = min(keep_len, remaining_context.shape[1])
                remaining_context = remaining_context[:, -keep_len:]
                break

            if latent_tokens is None:
                chunk_len = self._compression_chunk_tokens(
                    token_capacity=available_without_latent,
                    available_tokens=remaining_context.shape[1],
                )
                compress_context = remaining_context[:, :chunk_len]
                remaining_context = remaining_context[:, chunk_len:]
                compress_input = compress_context
            else:
                chunk_len = self._compression_chunk_tokens(
                    token_capacity=available_with_latent,
                    available_tokens=remaining_context.shape[1],
                )
                compress_context = remaining_context[:, :chunk_len]
                compress_input = torch.cat([latent_tokens, compress_context], dim=1)
                remaining_context = remaining_context[:, chunk_len:]

            new_latent = self._compress_sequence(compress_input, compression_round)

            latent_tokens = new_latent
            compression_round += 1

        if latent_tokens is not None:
            latent_with_type = latent_tokens + self.latent_type_embedding.expand(
                latent_tokens.shape[0], latent_tokens.shape[1], -1
            )
            full_input = torch.cat([latent_with_type, remaining_context, query_token], dim=1)
        else:
            full_input = torch.cat([remaining_context, query_token], dim=1)

        out, _ = self.transformer(full_input, return_attentions=False)
        compression_info["num_compressions"] = compression_round
        compression_info["has_latent_prefix"] = latent_tokens is not None
        compression_info["visible_context_tokens"] = remaining_context.shape[1]
        return out, compression_info

    def _select_action_positions(self, transformer_output, visible_context_tokens, has_latent_prefix):
        latent_offset = self.n_compress_tokens if has_latent_prefix else 0
        query_idx = latent_offset + visible_context_tokens
        visible_transitions = visible_context_tokens // self.tokens_per_transition
        state_positions = latent_offset + torch.arange(
            0,
            visible_transitions * self.tokens_per_transition,
            self.tokens_per_transition,
            device=transformer_output.device,
            dtype=torch.long,
        )
        all_positions = torch.cat(
            [state_positions, torch.tensor([query_idx], device=transformer_output.device, dtype=torch.long)],
            dim=0,
        )
        selected = transformer_output.index_select(1, all_positions)
        return self.pred_action(selected), visible_transitions

    def forward(self, x):
        query_states = x["query_states"].to(self.device)
        target_actions = x["target_actions"].to(self.device)
        states = x["states"].to(self.device)
        actions = x["actions"].to(self.device)
        rewards = x["rewards"].to(self.device)
        context_lengths = x.get("context_lengths", None)

        # Fast path for fixed-length contexts (AD-style batches).
        if context_lengths is None:
            context_tokens = self._build_context_tokens(states, actions, rewards)
            query_token = self._build_query_token(query_states)

            transformer_output, compression_info = self._forward_with_compression(
                context_tokens=context_tokens, query_token=query_token
            )

            logits, visible_transitions = self._select_action_positions(
                transformer_output=transformer_output,
                visible_context_tokens=compression_info["visible_context_tokens"],
                has_latent_prefix=compression_info["has_latent_prefix"],
            )

            start_idx = actions.shape[1] - visible_transitions
            action_targets = actions[:, start_idx:]
            target_actions_seq = torch.cat([action_targets, target_actions.unsqueeze(1)], dim=1)

            loss_action = self.loss_fn(
                logits.reshape(-1, logits.size(-1)),
                target_actions_seq.reshape(-1),
            )
            acc_action = (logits.argmax(dim=-1) == target_actions_seq).float().mean()
            loss_total = self._build_total_loss(loss_action)

            return {
                "loss_action": loss_action,
                "acc_action": acc_action,
                "loss_total": loss_total,
                "num_compressions": compression_info["num_compressions"],
                "attentions": None,
            }

        # Variable-length RAD batches (padded + context_lengths).
        # To avoid slow per-sample forward passes, process samples in groups
        # that share the same context length.
        context_lengths = context_lengths.to(self.device)
        if self.forward_context_bucket > 1:
            bucket = self.forward_context_bucket
            context_lengths = torch.where(
                context_lengths < bucket,
                context_lengths,
                (context_lengths // bucket) * bucket,
            )

        # Precompute token embeddings once for the padded batch, then slice per-group.
        state_ids_all = map_dark_states(states.to(torch.long), self.grid_size)
        state_tokens_all = self.embed_state(state_ids_all)
        action_tokens_all = self.embed_action(actions.to(torch.long))
        reward_tokens_all = self.embed_reward(rewards.unsqueeze(-1).to(torch.float))
        query_tokens_all = self._build_query_token(query_states)

        logits_flat = []
        targets_flat = []
        compression_sum = 0.0
        n_samples = 0

        unique_ctx_lengths = torch.unique(context_lengths, sorted=True)
        for ctx_len_t in unique_ctx_lengths:
            ctx_len = int(ctx_len_t.item())
            group_idx = (context_lengths == ctx_len_t).nonzero(as_tuple=False).squeeze(1)
            group_size = int(group_idx.numel())
            if group_size == 0:
                continue

            state_tokens_g = state_tokens_all.index_select(0, group_idx)[:, :ctx_len]
            action_tokens_g = action_tokens_all.index_select(0, group_idx)[:, :ctx_len]
            reward_tokens_g = reward_tokens_all.index_select(0, group_idx)[:, :ctx_len]
            query_token_g = query_tokens_all.index_select(0, group_idx)
            target_action_g = target_actions.index_select(0, group_idx)

            context_tokens_g = self._pack_sar_tokens(
                state_tokens=state_tokens_g,
                action_tokens=action_tokens_g,
                reward_tokens=reward_tokens_g,
            )

            transformer_output_g, compression_info_g = self._forward_with_compression(
                context_tokens=context_tokens_g, query_token=query_token_g
            )

            logits_g, visible_transitions_g = self._select_action_positions(
                transformer_output=transformer_output_g,
                visible_context_tokens=compression_info_g["visible_context_tokens"],
                has_latent_prefix=compression_info_g["has_latent_prefix"],
            )

            actions_g = actions.index_select(0, group_idx)[:, :ctx_len]
            start_idx = ctx_len - visible_transitions_g
            action_targets_g = actions_g[:, start_idx:]
            target_actions_seq_g = torch.cat([action_targets_g, target_action_g.unsqueeze(1)], dim=1)

            logits_flat.append(logits_g.reshape(-1, logits_g.size(-1)))
            targets_flat.append(target_actions_seq_g.reshape(-1))

            compression_sum += compression_info_g["num_compressions"] * group_size
            n_samples += group_size

        logits_flat = torch.cat(logits_flat, dim=0)
        targets_flat = torch.cat(targets_flat, dim=0)

        loss_action = self.loss_fn(logits_flat, targets_flat)
        acc_action = (logits_flat.argmax(dim=-1) == targets_flat).float().mean()
        avg_num_compressions = float(compression_sum / max(n_samples, 1))
        loss_total = self._build_total_loss(loss_action)

        return {
            "loss_action": loss_action,
            "acc_action": acc_action,
            "loss_total": loss_total,
            "num_compressions": avg_num_compressions,
            "attentions": None,
        }

    def _build_total_loss(self, loss_action):
        """
        Touch compression parameters with zero-weight terms so DDP can run with
        find_unused_parameters=False even on no-compression batches.
        """
        zero = self.latent_type_embedding.reshape(-1)[0] * 0.0
        for p in self._compression_params:
            zero = zero + p.reshape(-1)[0] * 0.0
        return loss_action + zero

    def evaluate_in_context(self, vec_env, eval_timesteps, beam_k=0, sample=True, return_attentions=False):
        outputs = {
            "reward_episode": [],
            "compression_events": [],
        }

        reward_episode = np.zeros(vec_env.num_envs)
        query_states = torch.tensor(
            vec_env.reset(), device=self.device, requires_grad=False, dtype=torch.long
        )

        query_token = self._build_query_token(query_states)
        latent_tokens = None
        transition_buffer = None
        compression_count = 0

        for step in range(eval_timesteps):
            query_states_prev = query_states.clone().detach()

            if latent_tokens is not None and transition_buffer is not None:
                transformer_input = torch.cat([latent_tokens, transition_buffer, query_token], dim=1)
            elif transition_buffer is not None:
                transformer_input = torch.cat([transition_buffer, query_token], dim=1)
            else:
                transformer_input = query_token

            output, _ = self.transformer(transformer_input, return_attentions=False)
            logits = self.pred_action(output[:, -1])

            if sample:
                log_probs = F.log_softmax(logits, dim=-1)
                actions = torch.multinomial(log_probs.exp(), num_samples=1).squeeze(1)
            else:
                actions = logits.argmax(dim=-1)

            query_states_np, rewards_np, dones, infos = vec_env.step(actions.cpu().numpy())

            reward_episode += rewards_np
            rewards = torch.tensor(
                rewards_np, device=self.device, requires_grad=False, dtype=torch.float
            )

            if dones[0]:
                outputs["reward_episode"].append(reward_episode)
                reward_episode = np.zeros(vec_env.num_envs)

            query_states = torch.tensor(
                query_states_np, device=self.device, requires_grad=False, dtype=torch.long
            )
            query_token = self._build_query_token(query_states)

            state_token = self.embed_state(map_dark_states(query_states_prev.to(torch.long), self.grid_size))
            action_token = self.embed_action(actions.to(torch.long))
            reward_token = self.embed_reward(rewards.unsqueeze(-1))
            new_transition = torch.stack([state_token, action_token, reward_token], dim=1)

            if transition_buffer is None:
                transition_buffer = new_transition
            else:
                transition_buffer = torch.cat([transition_buffer, new_transition], dim=1)

            max_buffer_len = self.max_seq_length - 1
            if latent_tokens is not None:
                max_buffer_len -= self.n_compress_tokens

            if transition_buffer.shape[1] > max_buffer_len:
                if latent_tokens is not None:
                    compress_input = torch.cat([latent_tokens, transition_buffer], dim=1)
                else:
                    compress_input = transition_buffer
                latent_tokens = self.compression_transformer(compress_input)
                latent_tokens = latent_tokens + self.latent_type_embedding.expand(
                    latent_tokens.shape[0], latent_tokens.shape[1], -1
                )
                transition_buffer = None
                compression_count += 1
                outputs["compression_events"].append(step)

        outputs["reward_episode"] = np.stack(outputs["reward_episode"], axis=1)
        outputs["total_compressions"] = compression_count
        return outputs

    def set_curriculum(self, max_compressions):
        self.max_compressions = max_compressions
