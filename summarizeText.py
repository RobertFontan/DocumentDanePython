import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, SafetySetting


def generate():
    vertexai.init(project="iconic-reactor-437022-u3", location="us-east4")
    model = GenerativeModel(
        "gemini-1.5-flash-002",
        system_instruction=["""Summarize the legal text in a friendly and simple way for people of any background to understand."""]
    )
    responses = model.generate_content(
        [text1],
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=True,
    )

    for response in responses:
        print(response.text, end="")

text1 = """NON-DISCLOSURE AGREEMENT PARTIES - This Non-Disclosure Agreement (hereinafter referred to as the “Agreement”) is entered into on ________________ (the “Effective Date”), by and between ________________________, with an address of ________________, (hereinafter referred to as the “Disclosing Party”) and ________________, with an address of ________________, (hereinafter referred to as the “Receiving Party”) (collectively referred to as the “Parties”). CONFIDENTIAL INFORMATION - The Receiving Party agrees not to disclose, copy, clone, or modify any confidential information related to the Disclosing Party and agrees not to use any such information without obtaining consent. - “Confidential information” refers to any data and/or information that is related to the Disclosing Party, in any form, including, but not limited to, oral or written. Such confidential information includes, but is not limited to, any information related to the business or industry of the Disclosing Party, such as discoveries, processes, techniques, programs, knowledge bases, customer lists, potential customers, business partners, affiliated partners, leads, knowhow, or any other services related to the Disclosing Party. RETURN OF CONFIDENTIAL INFORMATION - The Receiving Party agrees to return all the confidential information to the Disclosing Party upon the termination of this Agreement. OWNERSHIP - This Agreement is not transferable and may only be transferred by written consent provided by both Parties. GOVERNING LAW - This Agreement shall be governed by and construed in accordance with the laws of ________________. SIGNATURE AND DATE - The Parties hereby agree to the terms and conditions set forth in this Agreement and such is demonstrated by their signatures below: DISCLOSING PARTY Name:____________________________ Signature:_________________________ Date:_____________________________ RECEIVING PARTY Name:____________________________ Signature:_________________________ Date:_____________________________"""

generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}

safety_settings = [
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
]

generate()