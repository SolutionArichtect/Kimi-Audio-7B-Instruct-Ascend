from kimia_infer.api.kimia import KimiAudio
import os
import soundfile as sf

if __name__ == "__main__":

    model = KimiAudio(
        # 切换到本地路径进行加载
        model_path="/workspace/model/Kimi-Audio-7B-Instruct",
        load_detokenizer=True,
    )

    sampling_params = {
        "audio_temperature": 0.8,
        "audio_top_k": 10,
        "text_temperature": 0.0,
        "text_top_k": 5,
        "audio_repetition_penalty": 1.0,
        "audio_repetition_window_size": 64,
        "text_repetition_penalty": 1.0,
        "text_repetition_window_size": 16,
    }

    # 预热任务，确保模型加载
    model.generate(
        [{"role": "user", "message_type": "text", "content": "预热"},
         {"role": "user", "message_type": "audio", "content": "test_audios/asr_example.wav"}],
        **sampling_params,
        output_type="text"
    )

    # 列出所有音频文件用于音频转文本
    audio_files = [
        "test_audios/asr_example.wav",
        "test_audios/qa_example.wav",
        "test_audios/multiturn/case1/multiturn_a1.wav",
        "test_audios/multiturn/case2/multiturn_a1.wav"
    ]

    output_dir = "test_audios/output"
    os.makedirs(output_dir, exist_ok=True)

    # 遍历音频文件，进行音频转文本
    for audio_file in audio_files:
        messages = [
            {"role": "user", "message_type": "text", "content": "请将音频内容转换为文字。"},
            {"role": "user", "message_type": "audio", "content": audio_file},
        ]

        wav, text = model.generate(messages, **sampling_params, output_type="both")

        # 检查 wav 是否为 None
        if wav is not None:
            print(f">>> output text for {audio_file}: ", text)

            # 保存音频输出
            wav_output_path = os.path.join(output_dir, f"{os.path.basename(audio_file)}_output.wav")
            sf.write(
                wav_output_path,
                wav.detach().cpu().view(-1).numpy(),
                24000,
            )
            print(f">>> output audio saved to: {wav_output_path}")
        else:
            print(f">>> Error: No audio output for {audio_file}.")
