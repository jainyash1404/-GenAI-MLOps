import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")


def generate_recommendations(student_data, predicted_score):

    if not API_KEY:
        return "Gemini API key missing. Please check .env file."

    try:
        client = genai.Client(
            api_key=API_KEY,
            http_options=types.HttpOptions(timeout=12000)
        )

        prompt = f"""
        Act as a student academic mentor.

        Student Details:
        Attendance: {student_data['attendance']}%
        Study Hours: {student_data['study_hours']}
        Previous Score: {student_data['previous_score']}
        Assignment Completion: {student_data['assignment_completion']}%
        Sleep: {student_data['sleep_hours']} hours

        Predicted Score: {predicted_score}/100

        Give:
        1. Short performance analysis.
        2. Three weaknesses.
        3. Three improvement tips.
        4. A simple 7-day study plan.

        Keep everything concise and practical.
        """

        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=500,
                temperature=0.3
            )
        )

        return response.text or "No recommendations generated."

    except Exception as e:

        print("Gemini Error:", str(e))

        return """
        AI recommendations are temporarily unavailable.

        Suggested improvements:
        1. Maintain at least 85% attendance.
        2. Study for 3-4 hours daily.
        3. Complete all assignments on time.
        4. Revise previous topics regularly.
        5. Maintain a healthy sleep schedule.
        """