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
        gpt2_cfg._attn_implementation = "eager"
        self.transformer_model = GPT2Model(gpt2_cfg)

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

    def _forward_with_compression(self, context_tokens, query_token):
        context_len = context_tokens.shape[1]
        available_for_context = self.max_seq_length - 1

        compression_info = {
            "num_compressions": 0,
            "has_latent_prefix": False,
            "visible_context_tokens": context_len,
        }

        if context_len <= available_for_context:
            full_input = torch.cat([context_tokens, query_token], dim=1)
            out, _ = self.transformer(full_input, return_attentions=False)
            return out, compression_info

        latent_tokens = None
        remaining_context = context_tokens
        compression_round = 0

        while True:
            latent_len = self.n_compress_tokens if latent_tokens is not None else 0
            total_needed = latent_len + remaining_context.shape[1] + 1

            if total_needed <= self.max_seq_length:
                break

            if self.max_compressions is not None and compression_round >= self.max_compressions:
                keep_len = self.max_seq_length - latent_len - 1
                keep_len = self._round_down_to_transition_tokens(keep_len)
                keep_len = min(keep_len, remaining_context.shape[1])
                remaining_context = remaining_context[:, -keep_len:]
                break

            if latent_tokens is None:
                available_new = self.max_seq_length - 1
                overflow = remaining_context.shape[1] - available_new
                required = overflow + self.n_compress_tokens
                compress_context_len = self._round_up_to_transition_tokens(required)
                compress_context_len = min(compress_context_len, remaining_context.shape[1])
                compress_input = remaining_context[:, :compress_context_len]
                remaining_context = remaining_context[:, compress_context_len:]
            else:
                available_new = self.max_seq_length - self.n_compress_tokens - 1
                overflow = remaining_context.shape[1] - available_new
                compress_context_len = self._round_up_to_transition_tokens(overflow)
                compress_context_len = min(compress_context_len, remaining_context.shape[1])
                compress_context = remaining_context[:, :compress_context_len]
                compress_input = torch.cat([latent_tokens, compress_context], dim=1)
                remaining_context = remaining_context[:, compress_context_len:]

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

            return {
                "loss_action": loss_action,
                "acc_action": acc_action,
                "loss_total": loss_action,
                "num_compressions": compression_info["num_compressions"],
                "attentions": None,
            }

        # Variable-length RAD batches (padded + context_lengths).
        context_lengths = context_lengths.to(self.device)
        logits_flat = []
        targets_flat = []
        n_correct = 0
        n_total = 0
        compression_counts = []

        for i in range(states.shape[0]):
            ctx_len = int(context_lengths[i].item())

            states_i = states[i:i + 1, :ctx_len]
            actions_i = actions[i:i + 1, :ctx_len]
            rewards_i = rewards[i:i + 1, :ctx_len]
            query_i = query_states[i:i + 1]
            target_action_i = target_actions[i:i + 1]

            context_tokens_i = self._build_context_tokens(states_i, actions_i, rewards_i)
            query_token_i = self._build_query_token(query_i)

            transformer_output_i, compression_info_i = self._forward_with_compression(
                context_tokens=context_tokens_i, query_token=query_token_i
            )

            logits_i, visible_transitions_i = self._select_action_positions(
                transformer_output=transformer_output_i,
                visible_context_tokens=compression_info_i["visible_context_tokens"],
                has_latent_prefix=compression_info_i["has_latent_prefix"],
            )

            start_idx = actions_i.shape[1] - visible_transitions_i
            action_targets_i = actions_i[:, start_idx:]
            target_actions_seq_i = torch.cat([action_targets_i, target_action_i.unsqueeze(1)], dim=1)

            pred_i = logits_i.argmax(dim=-1)
            n_correct += (pred_i == target_actions_seq_i).sum().item()
            n_total += target_actions_seq_i.numel()

            logits_flat.append(logits_i.reshape(-1, logits_i.size(-1)))
            targets_flat.append(target_actions_seq_i.reshape(-1))
            compression_counts.append(compression_info_i["num_compressions"])

        logits_flat = torch.cat(logits_flat, dim=0)
        targets_flat = torch.cat(targets_flat, dim=0)

        loss_action = self.loss_fn(logits_flat, targets_flat)
        acc_action = torch.tensor(n_correct / max(n_total, 1), device=self.device, dtype=torch.float)
        avg_num_compressions = float(np.mean(compression_counts)) if compression_counts else 0.0

        return {
            "loss_action": loss_action,
            "acc_action": acc_action,
            "loss_total": loss_action,
            "num_compressions": avg_num_compressions,
            "attentions": None,
        }

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
