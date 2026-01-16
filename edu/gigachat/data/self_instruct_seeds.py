"""
Seed tasks for Self-Instruct pipeline.
These 20 manually generated tasks serve as the starting point for synthesizing new instructions.
"""

SEED_TASKS = [
    {
        "instruction": "Explain the concept of recursion to a 5-year-old.",
        "input": "",
        "output": "Imagine you have a big box of toys. You open it, and inside is a smaller box. You open that one, and there's an even smaller box! You keep opening boxes until you find a tiny candy at the end. Recursion is like that—solving a big problem by doing the same small thing over and over again until you're done!"
    },
    {
        "instruction": "Write a Python function to check if a string is a palindrome.",
        "input": "",
        "output": "```python\ndef is_palindrome(s):\n    # Remove non-alphanumeric characters and convert to lowercase\n    clean_s = ''.join(c.lower() for c in s if c.isalnum())\n    return clean_s == clean_s[::-1]\n```"
    },
    {
        "instruction": "Compose a haiku about a robot learning to love.",
        "input": "",
        "output": "Cold steel feels a beat,\nCircuits warm with new desire,\nHeart of code now wakes."
    },
    {
        "instruction": "Classify the following sentiment as positive, negative, or neutral.",
        "input": "The movie was okay, but the popcorn was stale.",
        "output": "Negative (leaning neutral, but the stale popcorn complaint drags it down)."
    },
    {
        "instruction": "List 5 creative uses for a brick.",
        "input": "",
        "output": "1. A bookend for heavy encyclopedias.\n2. A persistent doorstop.\n3. A canvas for painting a mini-landscape.\n4. A weight for pressing tofu.\n5. A heated bed warmer (wrapped in a towel)."
    },
    {
        "instruction": "Translate the following sentence into French.",
        "input": "The cat is sleeping on the table.",
        "output": "Le chat dort sur la table."
    },
    {
        "instruction": "Summarize the following text in one sentence.",
        "input": "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy that, through cellular respiration, can later be released to fuel the organism's activities. This chemical energy is stored in carbohydrate molecules, such as sugars, which are synthesized from carbon dioxide and water.",
        "output": "Photosynthesis is the process by which plants convert light energy into chemical energy stored as sugar."
    },
    {
        "instruction": "Debug this code snippet.",
        "input": "for i in ranges(10)\n    print(i)",
        "output": "There are two errors: `ranges` should be `range`, and a colon is missing after the parenthesis.\n\nCorrected generic code:\n```python\nfor i in range(10):\n    print(i)\n```"
    },
    {
        "instruction": "What are the three primary colors in painting?",
        "input": "",
        "output": "Red, Blue, and Yellow."
    },
    {
        "instruction": "Create a JSON object representing a user profile.",
        "input": "",
        "output": "{\n  \"id\": 101,\n  \"username\": \"coder_ace\",\n  \"email\": \"ace@example.com\",\n  \"is_active\": true,\n  \"roles\": [\"admin\", \"editor\"]\n}"
    },
    {
        "instruction": "Explain why the sky is blue.",
        "input": "",
        "output": "The sky appears blue because of Rayleigh scattering. As sunlight reaches Earth's atmosphere, it collides with gas molecules. Blue light has a shorter wavelength and is scattered in all directions more strongly than other colors, making the sky look blue to our eyes."
    },
    {
        "instruction": "Write a short email to a colleague rescheduling a meeting.",
        "input": "",
        "output": "Subject: Rescheduling our 2pm sync\n\nHi [Name],\n\nI'm afraid something urgent has came up and I won't be able to make our 2pm meeting today. Could we push it to tomorrow at 10am instead?\n\nThanks,\n[Your Name]"
    },
    {
        "instruction": "Convert the following temperature from Celsius to Fahrenheit.",
        "input": "25 degrees Celsius",
        "output": "77 degrees Fahrenheit (Formula: 25 * 9/5 + 32 = 77)."
    },
    {
        "instruction": "Identify the capital of Japan.",
        "input": "",
        "output": "Tokyo."
    },
    {
        "instruction": "Suggest a name for a new coffee shop that doubles as a bookstore.",
        "input": "",
        "output": "\"Brewed Awakenings\" or \"The Caffeinated Chapter\"."
    },
    {
        "instruction": "Solve for x: 2x + 5 = 13",
        "input": "",
        "output": "Subtract 5 from both sides: 2x = 8.\nDivide by 2: x = 4."
    },
    {
        "instruction": "Describe the main function of the mitochondria.",
        "input": "",
        "output": "The mitochondria is often called the powerhouse of the cell because its main function is to generate energy (ATP) through cellular respiration."
    },
    {
        "instruction": "Write a SQL query to select all users from California.",
        "input": "",
        "output": "```sql\nSELECT * FROM users WHERE state = 'CA';\n```"
    },
    {
        "instruction": "What is the next number in the sequence?",
        "input": "2, 4, 8, 16, ...",
        "output": "32 (each number is multiplied by 2)."
    },
    {
        "instruction": "Give me a recipe for a simple peanut butter and jelly sandwich.",
        "input": "",
        "output": "1. Take two slices of bread.\n2. Spread peanut butter on one slice.\n3. Spread jelly on the other slice.\n4. Press the two slices together.\n5. Enjoy!"
    }
]
