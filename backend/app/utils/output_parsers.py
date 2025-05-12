from typing import List
from langchain_core.output_parsers import BaseOutputParser

class LineListOutputParser(BaseOutputParser[List[str]]):
    """
    Output parser that converts newline-separated text into a list of strings.
    Used primarily to process multiple query variations from an LLM.
    """

    def parse(self, text: str) -> List[str]:
        """
        Parse the output text into a list of strings, one per line.
        
        Args:
            text: The text to parse
            
        Returns:
            A list of strings, with empty lines removed
        """
        lines = text.strip().split("\n")
        return list(filter(None, lines))  # Remove empty lines