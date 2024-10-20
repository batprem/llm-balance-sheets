from typing_extensions import TypedDict
from src.constants import MODEL
from openai import OpenAI


class ChatTurn(TypedDict):
    role: str
    content: str


def get_completion(
    prompt: str,
    client: OpenAI,
    model: str = MODEL,
    stream: bool = False,
    initial_message: list[ChatTurn] | None = None,
) -> str:
    """
    This function generates a completion for a given prompt using the OpenAI API.

    Arguments:
    - prompt (str): The input prompt for the completion.
    - client (OpenAI): An instance of the OpenAI API client.
    - model (str): The model to use for generating the completion. Default is MODEL constant.
    - stream (bool): Whether to stream the completion. Default is False.
    - initial_message (list[ChatTurn] | None): Initial messages to include in the conversation. Default is None.

    Returns:
    - str: The generated completion.
    """ # noqa E501
    if initial_message is None:
        initial_message = []
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=0, stream=stream
    )
    if not stream:
        return response.choices[0].message.content
    else:
        result = ""
        for chunk in response:
            content = chunk.choices[0].delta.content
            print(content, end="")
            if isinstance(content, str):
                result += content
        return result
