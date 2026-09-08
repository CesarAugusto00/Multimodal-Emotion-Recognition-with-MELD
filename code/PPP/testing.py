from pipeline.emotion_pipeline import EmotionPipeline
from pipeline.response_generator import ResponseGenerator


def main():
    # Load models
    emotion_pipeline = EmotionPipeline(
        checkpoint_path="Model/mgif_emotion_model.pt"
    )

    response_generator = ResponseGenerator(
        model_name="llama3.2:1b"
    )

    # Example MELD-style input
    user_text = "Put text here"

    #put path of audio file here
    audio_path = "test_data/example.mp4"

    # 1. Emotion recognition
    emotion_result = emotion_pipeline.predict(
        text=user_text,
        audio_path=audio_path
    )

    print("\nEmotion recognition result:")
    print(emotion_result)

    # 2. Generate robot response
    robot_response = response_generator.generate(
        emotion=emotion_result["emotion"],
        confidence=emotion_result["confidence"],
        user_text=user_text,
        conversation_history=[]
    )

    print("\nRobot response:")
    print(robot_response)

    # 3. Final structured output
    final_result = {
        "emotion": emotion_result["emotion"],
        "confidence": emotion_result["confidence"],
        "response": robot_response
    }

    print("\nFinal result:")
    print(final_result)


if __name__ == "__main__":
    main()