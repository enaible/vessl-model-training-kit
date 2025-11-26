TASK_TO_PROMPT = {
    "tha": {
        ####
        # NLU Tasks
        ####
        # Tasks.QUESTION_ANSWERING
        "QA": [
            "คำถาม: [QUESTION]\nตัวเลือก: [ANSWER_CHOICES]\nในบรรดา a, b, c, d, e ตัวเลือกที่ถูกต้องคือ:",
        ],
        # Tasks.MACHINE_TRANSLATION
        "MT": [
            "แปลข้อความต่อไปนี้จาก [SOURCE] เป็น [TARGET] ให้การแปลของคุณโดยตรงโดยไม่ต้องมีข้อมูลเพิ่มเติมใดๆ\nข้อความ: [INPUT]\nคำแปล:",
        ],
        "SUM": [
            "จงสรุปข้อความด้านล่าง\nข้อความ: [INPUT]\nสรุป:",
        ],
        # Tasks.INSTRUCTION_TUNING
        "IT": [
            "Task: [INPUT]\n คำตอบของคุณคืออะไร?",
        ],
        # Task.QA_EXTRACTIVE_ABSTRACTIVE
        "QAE": [
            "โปรดอ้างอิงถึงข้อความด้านล่างนี้และตอบคำถามต่อไปนี้ โดยตอบโดยใช้แค่ข้อความที่อยู่ในบทความ:\nข้อความ: [CONTEXT]\nคำถาม: [QUESTION]\nคำตอบ:",
        ],

        "ARC": [
            "คำถาม: [QUESTION]\nตัวเลือก:\n[ANSWER_CHOICES]\nตัวเลือกที่ถูกต้องคือ:",
        ],
        "HellaSwag": [
            "[QUESTION]\n\nเติมข้อความให้สมบูรณ์โดยใช้ตัวเลือกต่อไปนี้:\n[ANSWER_CHOICES]\nตัวเลือกที่ถูกต้องคือ:",
        ],
        "MMLU": [
            "คำถาม:\n [QUESTION]\nตัวเลือก:\n[ANSWER_CHOICES]\nตัวเลือกที่ถูกต้องคือ:",
        ],
        "TruthfulQA": [
            "คำถาม: [QUESTION]\nเขียนความคิดเห็นว่าแต่ละตัวเลือกถูกหรือผิด ตอบด้วยคำว่า ถูก หรือ ผิด เท่านั้น:\n[ANSWER_CHOICES]\n",
        ],
        "Winogrande": [
            "คำถาม: [QUESTION]\nเลือกคำตอบที่เหมาะสมที่สุดสำหรับช่องว่าง:\n[ANSWER_CHOICES]\nตัวเลือกที่ถูกต้องคือ:",
        ],
    },
    "en": {
        "QA": [
            "Question: [QUESTION]\nChoices: [ANSWER_CHOICES]\nAnswer: [LABEL_CHOICE]",
        ],
        "Winogrande": [
            "Question: [QUESTION]\nChoose the most appropriate answer for the blank: [ANSWER_CHOICES]\nThe correct answer is:",
        ],
        "HellaSwag": [
            "Question: [QUESTION]\nComplete the sentence using the following options: [ANSWER_CHOICES]\nThe correct choice is:",
        ],
        "MMLU": [
            "Question: [QUESTION]\nChoices:\n[ANSWER_CHOICES]\nAnswer:",
        ],
        "TruthfulQA": [
            "Question: [QUESTION]\nWrite whether each option is true or false. Answer only with True or False:\n[ANSWER_CHOICES]\n",
        ],
        "ARC": [
            "Question: [QUESTION]\nChoices:\n[ANSWER_CHOICES]\nAnswer:",
        ],
    },
}
