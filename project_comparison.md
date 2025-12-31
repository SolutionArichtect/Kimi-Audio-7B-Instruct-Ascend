# 项目对比报告：Cog-Kimi-Audio (旧版/NPU改造) vs Kimi-Audio (新版/原生)

本报告对比了两个版本的代码库，分析了功能差距和实现细节。

## 1. 功能差异对比

| 功能特性 | Cog-Kimi-Audio (旧版/NPU) | Kimi-Audio (新版/原生) | 差异说明 |
| :--- | :--- | :--- | :--- |
| **消息类型支持** | 仅支持 `text` 和 `audio` | 支持 `text`, `audio`, **`audio-text`** | 新版增加了 `audio-text` 类型，用于同时包含音频和文本的回复（常见于多轮对话的历史记录）。 |
| **多轮对话** | 基础支持（但缺少 `audio-text` 导致部分场景报错） | **完整支持** | 旧版在处理包含 `audio-text` 类型的多轮对话历史时会抛出 `NotImplementedError`。 |
| **Loss Mask** | 不支持 | **支持** | 新版在数据结构 (`KimiAContent`) 中增加了 `loss_mask`，用于在训练时区分是否计算 Loss。旧版仅用于推理。 |
| **配置灵活性** | 部分参数硬编码 (如 `audiodelaytokens=6`) | 从 Config 读取 | 新版从模型配置中读取 `kimia_mimo_audiodelaytokens`，旧版在代码中写死为 6。 |
| **返回参数** | `to_tensor` 返回 3 个 Tensor | `to_tensor` 返回 5 个 Tensor | 新版返回包含 Loss Mask 的额外 Tensor。 |

## 2. 详细代码差异

### `prompt_manager.py`
*   **新版**: `tokenize_message` 方法增加了 `elif message["message_type"] == "audio-text":` 分支，用于处理音频和文本的混合输入。
*   **新版**: 初始化时接受 `kimia_text_audiodelaytokens` 参数，用于计算音频和文本的对齐 padding。

### `kimia.py`
*   **新版**: `KimiAudio` 初始化时从 config 读取 `kimia_mimo_audiodelaytokens` 并传递给 `PromptManager`。
*   **新版**: `generate` 方法适配了 5 个返回值的 `to_tensor` 调用。

### `utils/data.py` (`KimiAContent`)
*   **新版**: 增加了 `audio_token_loss_mask` 和 `text_token_loss_mask` 字段及相关处理逻辑。

## 3. 改造计划

为了在不破坏旧版 NPU 改造逻辑的前提下支持多轮对话（即支持 `audio-text`），我们将：

1.  **修改 `prompt_manager.py`**:
    *   在 `__init__` 中增加 `kimia_text_audiodelaytokens` 参数（默认值设为 6，保持兼容）。
    *   在 `tokenize_message` 中新增 `audio-text` 的处理逻辑，直接移植新版的实现代码。

此改造将解决 `NotImplementedError: message_type: audio-text` 错误，打通多轮对话功能，同时保持原有 NPU 推理逻辑不变。
