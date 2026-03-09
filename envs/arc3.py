from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np


def _to_nested_list(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if hasattr(value, "value"):
            return int(value.value)
        return int(value)
    except Exception:
        return int(default)


def _extract_action_context(action_input: Any) -> tuple[int, dict[str, Any]]:
    if action_input is None:
        return 0, {}
    action_id = _to_int(getattr(action_input, "id", 0), default=0)
    data = getattr(action_input, "data", {}) or {}
    if hasattr(data, "model_dump"):
        data = data.model_dump()
    if not isinstance(data, dict):
        data = {}
    return action_id, data


def _normalize_frame_list(frame_any: Any) -> list[list[Any]]:
    frame = _to_nested_list(frame_any)
    for _ in range(3):
        if not isinstance(frame, list) or not frame:
            break
        first = _to_nested_list(frame[0])
        if isinstance(first, list) and first and isinstance(_to_nested_list(first[0]), list):
            frame = first
            continue
        break
    return frame if isinstance(frame, list) else []


def normalize_arc_frame(frame_any: Any, size: tuple[int, int], pad_value: int = 0) -> np.ndarray:
    frame = _normalize_frame_list(frame_any)
    height, width = map(int, size)
    grid = np.full((height, width), int(pad_value), dtype=np.int32)
    for y, row in enumerate(frame[:height]):
        row = _to_nested_list(row)
        if not isinstance(row, list):
            continue
        for x, value in enumerate(row[:width]):
            try:
                grid[y, x] = int(value)
            except Exception:
                grid[y, x] = int(pad_value)
    return grid


def extract_arc_frame_metadata(frame_any: Any, size: tuple[int, int], pad_value: int = 0) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    frame = _normalize_frame_list(frame_any)
    height, width = map(int, size)
    actual_h = min(len(frame), height)
    actual_w = 0
    for row in frame[:actual_h]:
        row = _to_nested_list(row)
        if isinstance(row, list):
            actual_w = max(actual_w, min(len(row), width))

    grid = np.full((height, width), int(pad_value), dtype=np.int32)
    valid_mask = np.zeros((height, width), dtype=np.float32)
    for y, row in enumerate(frame[:height]):
        row = _to_nested_list(row)
        if not isinstance(row, list):
            continue
        for x, value in enumerate(row[:width]):
            valid_mask[y, x] = 1.0
            try:
                grid[y, x] = int(value)
            except Exception:
                grid[y, x] = int(pad_value)
    return grid, valid_mask, (actual_h, actual_w)


def encode_arc_grid(grid: np.ndarray, num_colors: int, encoding: str, num_special_tokens: int = 1) -> np.ndarray:
    grid = np.asarray(grid, dtype=np.int32)
    vocab_size = int(num_colors) + int(num_special_tokens)
    grid = np.clip(grid, 0, vocab_size - 1)
    if encoding == "token":
        return grid[..., None].astype(np.int32)
    if encoding == "onehot":
        return np.eye(vocab_size, dtype=np.float32)[grid]
    if encoding == "scalar":
        denom = max(1, vocab_size - 1)
        return (grid.astype(np.float32) / float(denom))[..., None]
    raise ValueError(f"Unsupported ARC3 grid encoding: {encoding}")


def encode_state_flags(state_name: str) -> np.ndarray:
    state = str(state_name).upper()
    return np.array(
        [
            state == "NOT_PLAYED",
            state not in {"NOT_PLAYED", "WIN", "GAME_OVER"},
            state == "WIN",
            state == "GAME_OVER",
        ],
        dtype=np.float32,
    )


def encode_arc_state_flags(state_name: str) -> np.ndarray:
    return encode_state_flags(state_name)


def encode_arc_action_context(
    *,
    full_reset: bool,
    action_id: int,
    action_data: dict[str, Any],
    action_count: int,
    size: tuple[int, int],
) -> np.ndarray:
    has_coords = "x" in action_data and "y" in action_data
    width = max(1, int(size[1]) - 1)
    height = max(1, int(size[0]) - 1)
    x = float(_to_int(action_data.get("x", 0), default=0)) / float(width)
    y = float(_to_int(action_data.get("y", 0), default=0)) / float(height)
    return np.array(
        [
            float(bool(full_reset)),
            float(int(action_id)) / float(max(1, int(action_count) - 1)),
            float(int(action_id) == 0),
            float(int(action_id) == 5),
            float(int(action_id) == 6),
            float(bool(has_coords)),
            float(np.clip(x, 0.0, 1.0)),
            float(np.clip(y, 0.0, 1.0)),
        ],
        dtype=np.float32,
    )


def encode_arc_progress(
    *,
    levels_completed: int,
    win_levels: int,
    available_actions: list[int],
    action_count: int,
    frame_shape: tuple[int, int],
    size: tuple[int, int],
) -> np.ndarray:
    progress_ratio = float(levels_completed) / float(win_levels) if win_levels > 0 else 0.0
    remaining_ratio = float(max(0, win_levels - levels_completed)) / float(max(1, win_levels))
    action_ratio = float(len(available_actions)) / float(max(1, action_count))
    frame_h = float(max(0, int(frame_shape[0]))) / float(max(1, int(size[0])))
    frame_w = float(max(0, int(frame_shape[1]))) / float(max(1, int(size[1])))
    area = float(max(0, int(frame_shape[0]) * int(frame_shape[1]))) / float(max(1, int(size[0]) * int(size[1])))
    return np.array(
        [
            float(levels_completed),
            float(win_levels),
            float(progress_ratio),
            float(remaining_ratio),
            float(action_ratio),
            float(np.clip(frame_h, 0.0, 1.0)),
            float(np.clip(frame_w, 0.0, 1.0)),
            float(np.clip(area, 0.0, 1.0)),
        ],
        dtype=np.float32,
    )


def encode_action_mask(available_actions: list[int], action_count: int) -> np.ndarray:
    mask = np.zeros(int(action_count), dtype=np.float32)
    for value in available_actions:
        if 0 <= int(value) < int(action_count):
            mask[int(value)] = 1.0
    return mask


def decode_arc_action(action: np.ndarray, action_count: int, width: int, height: int) -> tuple[int, int, int]:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    split0 = int(action_count)
    split1 = split0 + int(width)
    expected = split1 + int(height)
    if action.size != expected:
        raise ValueError(f"ARC3 action size mismatch: expected {expected}, got {action.size}.")
    action_id = int(np.argmax(action[:split0]))
    x = int(np.argmax(action[split0:split1]))
    y = int(np.argmax(action[split1:]))
    return action_id, x, y


def derive_arc_reward(
    prev_levels_completed: int,
    prev_state_name: str,
    next_levels_completed: int,
    next_state_name: str,
    *,
    reward_per_level: float,
    reward_win: float,
    reward_loss: float,
) -> float:
    reward = float(next_levels_completed - prev_levels_completed) * float(reward_per_level)
    prev_state = str(prev_state_name).upper()
    next_state = str(next_state_name).upper()
    if next_state == "WIN" and prev_state != "WIN":
        reward += float(reward_win)
    if next_state == "GAME_OVER" and prev_state != "GAME_OVER":
        reward += float(reward_loss)
    return reward


@dataclass
class Arc3Transition:
    grid: np.ndarray
    valid_mask: np.ndarray
    frame_shape: tuple[int, int]
    state_name: str
    levels_completed: int
    win_levels: int
    available_actions: list[int]
    game_id: str
    guid: str
    full_reset: bool
    action_id: int
    action_data: dict[str, Any]
    raw: Any


class Arc3ActionSpace(gym.spaces.Box):
    def __init__(self, action_count: int, width: int, height: int):
        super().__init__(0.0, 1.0, shape=(int(action_count), int(width), int(height)), dtype=np.float32)
        self.action_count = int(action_count)
        self.width = int(width)
        self.height = int(height)
        self.multi_discrete = True

    def sample(self, mask=None, probability=None):
        action = np.zeros(self.action_count + self.width + self.height, dtype=np.float32)
        action[np.random.randint(self.action_count)] = 1.0
        action[self.action_count + np.random.randint(self.width)] = 1.0
        action[self.action_count + self.width + np.random.randint(self.height)] = 1.0
        return action


class Arc3Grid(gym.Env):
    metadata = {}

    def __init__(
        self,
        game_id: str,
        size: tuple[int, int] = (64, 64),
        grid_encoding: str = "token",
        num_colors: int = 16,
        num_special_tokens: int = 1,
        reward_per_level: float = 1.0,
        reward_win: float = 10.0,
        reward_loss: float = 0.0,
        operation_mode: str = "offline",
        environments_dir: str = "environment_files",
        recordings_dir: str = "recordings",
        arc_api_key: str = "",
        arc_base_url: str = "https://three.arcprize.org",
        seed: int = 0,
    ):
        super().__init__()
        self._game_id = str(game_id)
        self._size = tuple(map(int, size))
        self._grid_encoding = str(grid_encoding)
        self._num_colors = int(num_colors)
        self._num_special_tokens = max(1, int(num_special_tokens))
        self._pad_token_id = self._num_colors
        self._reward_per_level = float(reward_per_level)
        self._reward_win = float(reward_win)
        self._reward_loss = float(reward_loss)
        self._seed = int(seed)
        self._action_count = 8
        self._arc = self._make_arcade(
            arc_api_key=arc_api_key,
            arc_base_url=arc_base_url,
            operation_mode=operation_mode,
            environments_dir=environments_dir,
            recordings_dir=recordings_dir,
        )
        self._GameAction = self._import_game_action()
        self._env = None
        self._last_transition = None
        self.reward_range = [-np.inf, np.inf]

        if self._grid_encoding == "token":
            grid_space = gym.spaces.Box(0, self._num_colors + self._num_special_tokens - 1, self._size + (1,), dtype=np.int32)
        else:
            grid_channels = (self._num_colors + self._num_special_tokens) if self._grid_encoding == "onehot" else 1
            grid_space = gym.spaces.Box(0.0, 1.0, self._size + (grid_channels,), dtype=np.float32)
        self._observation_space = gym.spaces.Dict(
            {
                "grid": grid_space,
                "state_flags": gym.spaces.Box(0.0, 1.0, (4,), dtype=np.float32),
                "progress": gym.spaces.Box(-np.inf, np.inf, (8,), dtype=np.float32),
                "action_context": gym.spaces.Box(-np.inf, np.inf, (8,), dtype=np.float32),
                "action_mask": gym.spaces.Box(0.0, 1.0, (self._action_count,), dtype=np.float32),
                "is_first": gym.spaces.Box(0, 1, (), bool),
                "is_last": gym.spaces.Box(0, 1, (), bool),
                "is_terminal": gym.spaces.Box(0, 1, (), bool),
            }
        )
        self._action_space = Arc3ActionSpace(self._action_count, self._size[1], self._size[0])

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def reset(self):
        self._env = self._arc.make(self._game_id)
        if self._env is None:
            raise RuntimeError(
                f"ARC3 environment '{self._game_id}' could not be initialized. "
                "Check Python >= 3.12, arc-agi installation, OPERATION_MODE, and environment files."
            )
        transition = self._read_transition(self._env.observation_space)
        if transition.state_name.upper() in {"NOT_PLAYED", "GAME_OVER"}:
            transition = self._apply_action(0, 0, 0)
        self._last_transition = transition
        return self._format_obs(transition, is_first=True, is_last=False, is_terminal=False)

    def step(self, action):
        if self._env is None or self._last_transition is None:
            raise RuntimeError("ARC3 environment must be reset before stepping.")
        action_id, x, y = decode_arc_action(action, self._action_count, self._size[1], self._size[0])
        available_actions = self._last_transition.available_actions
        invalid_action = bool(available_actions) and int(action_id) not in set(map(int, available_actions))
        if invalid_action:
            action_id = int(sorted(set(map(int, available_actions)))[0])
        transition = self._apply_action(action_id, x, y)
        reward = derive_arc_reward(
            self._last_transition.levels_completed,
            self._last_transition.state_name,
            transition.levels_completed,
            transition.state_name,
            reward_per_level=self._reward_per_level,
            reward_win=self._reward_win,
            reward_loss=self._reward_loss,
        )
        done = transition.state_name.upper() in {"WIN", "GAME_OVER"}
        obs = self._format_obs(transition, is_first=False, is_last=done, is_terminal=done)
        info = {
            "levels_completed": int(transition.levels_completed),
            "win_levels": int(transition.win_levels),
            "available_actions": list(map(int, transition.available_actions)),
            "invalid_action": invalid_action,
            "game_id": transition.game_id,
            "guid": transition.guid,
            "full_reset": bool(transition.full_reset),
            "action_id": int(transition.action_id),
            "action_data": dict(transition.action_data),
            "frame_shape": tuple(int(v) for v in transition.frame_shape),
        }
        self._last_transition = transition
        return obs, reward, done, info

    def close(self):
        self._env = None

    def _make_arcade(
        self,
        *,
        arc_api_key: str,
        arc_base_url: str,
        operation_mode: str,
        environments_dir: str,
        recordings_dir: str,
    ):
        try:
            from arc_agi import Arcade, OperationMode
        except Exception as exc:
            raise RuntimeError(
                "ARC3 integration requires the 'arc-agi' package and its runtime dependencies."
            ) from exc

        raw_mode = str(operation_mode).strip().lower()
        try:
            mode = OperationMode(raw_mode)
        except Exception:
            mode = OperationMode.NORMAL
        return Arcade(
            arc_api_key=arc_api_key,
            arc_base_url=arc_base_url,
            operation_mode=mode,
            environments_dir=environments_dir,
            recordings_dir=recordings_dir,
        )

    def _import_game_action(self):
        try:
            from arcengine import GameAction
        except Exception as exc:
            raise RuntimeError("ARC3 integration requires the 'arcengine' package.") from exc
        return GameAction

    def _build_action(self, action_id: int, x: int, y: int):
        action = self._GameAction.from_id(int(action_id))
        if hasattr(action, "is_complex") and action.is_complex():
            action.set_data({"x": int(x), "y": int(y)})
        return action

    def _apply_action(self, action_id: int, x: int, y: int) -> Arc3Transition:
        action = self._build_action(action_id, x, y)
        data = action.action_data.model_dump() if hasattr(action, "action_data") else {}
        raw = self._env.step(action, data=data, reasoning={})
        return self._read_transition(raw)

    def _read_transition(self, raw: Any) -> Arc3Transition:
        if raw is None:
            raise ValueError("ARC3 environment returned no frame data.")
        state_obj = getattr(raw, "state", None)
        state_name = getattr(state_obj, "name", str(state_obj))
        available_actions = [int(v) for v in getattr(raw, "available_actions", [])]
        action_id, action_data = _extract_action_context(getattr(raw, "action_input", None))
        grid, valid_mask, frame_shape = extract_arc_frame_metadata(
            getattr(raw, "frame", []),
            self._size,
            pad_value=self._pad_token_id,
        )
        transition = Arc3Transition(
            grid=grid,
            valid_mask=valid_mask,
            frame_shape=frame_shape,
            state_name=str(state_name),
            levels_completed=int(getattr(raw, "levels_completed", 0)),
            win_levels=int(getattr(raw, "win_levels", 0)),
            available_actions=available_actions,
            game_id=str(getattr(raw, "game_id", self._game_id)),
            guid=str(getattr(raw, "guid", "")),
            full_reset=bool(getattr(raw, "full_reset", False)),
            action_id=action_id,
            action_data=action_data,
            raw=raw,
        )
        return transition

    def _format_obs(self, transition: Arc3Transition, *, is_first: bool, is_last: bool, is_terminal: bool):
        return {
            "grid": encode_arc_grid(
                transition.grid,
                self._num_colors,
                self._grid_encoding,
                num_special_tokens=self._num_special_tokens,
            ),
            "state_flags": encode_arc_state_flags(transition.state_name),
            "progress": encode_arc_progress(
                levels_completed=transition.levels_completed,
                win_levels=transition.win_levels,
                available_actions=transition.available_actions,
                action_count=self._action_count,
                frame_shape=transition.frame_shape,
                size=self._size,
            ),
            "action_context": encode_arc_action_context(
                full_reset=transition.full_reset,
                action_id=transition.action_id,
                action_data=transition.action_data,
                action_count=self._action_count,
                size=self._size,
            ),
            "action_mask": encode_action_mask(transition.available_actions, self._action_count),
            "is_first": bool(is_first),
            "is_last": bool(is_last),
            "is_terminal": bool(is_terminal),
        }
