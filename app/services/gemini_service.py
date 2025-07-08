from google import genai
from google.genai import types


class GeminiService:
    """
    Service for interacting with Google Gemini API.
    This service is used to generate content using the Gemini model.
    """

    def __init__(self, api_key: str):
        """
        Initializes the GeminiService with the provided API key.
        """
        self.client = genai.Client(api_key=api_key)

    def config_web_search_tool(self):
        """
        Configures the web search tool for grounding content generation.
        This tool allows the model to access real-time information from the web.
        """
        self.grounding_tool = types.Tool(
            google_search=types.GoogleSearch()
        )
        self.config = types.GenerateContentConfig(
            tools=[self.grounding_tool]
        )

    def get_completion(self, query: str, web_search: bool = True, model: str = "gemini-2.5-flash"):
        """
        Generates a response for the given query using the specified model.

        Args:
            query (str): The input query for which to generate content.
            web_search (bool): Whether to use web search for grounding. Defaults to True.
            model (str): The model to use for content generation. Defaults to "gemini-2.5-flash".
        Returns:
            The Google Gemini response object containing the generated content.
            Refer to https://ai.google.dev/gemini-api/docs/google-search for more details.
        """
        if web_search:
            self.config_web_search_tool()

        response = self.client.models.generate_content(
            model=model,
            contents=query,
            config=self.config,
        )
        return response

    def full_citations(self, response):
        """
        Adds citations to the response text based on grounding metadata.
        Adds a list of unique citations titles to the response as well.
        Refer to https://ai.google.dev/gemini-api/docs/google-search for more details.

        Args:
            response: The response object from the Gemini API containing grounding metadata.
        Returns:
            A dictionary containing the modified text with citations and a list of unique citation titles.
        """
        text = response.text
        supports = response.candidates[0].grounding_metadata.grounding_supports
        chunks = response.candidates[0].grounding_metadata.grounding_chunks

        # Sort supports by end_index in descending order to avoid shifting issues when inserting.
        sorted_supports = sorted(supports, key=lambda s: s.segment.end_index, reverse=True)

        list_citations: list[str] = []

        for support in sorted_supports:
            end_index = support.segment.end_index
            if support.grounding_chunk_indices:
                # Create citation string like [1](link1)[2](link2)
                citation_links = []
                for i in support.grounding_chunk_indices:
                    if i < len(chunks):
                        title = chunks[i].web.title
                        citation_links.append(f"[{i + 1}]({title})")
                        
                        # Add to full list of citations
                        if title not in list_citations:
                            list_citations.append(title)

                citation_string = ", ".join(citation_links)
                text = text[:end_index] + citation_string + text[end_index:]

        return {"text": text, "citations": list_citations}
    
    def list_citations(self, response):
        """
        Extracts a list of unique citation titles from the response.

        Args:
            response: The response object from the Gemini API containing grounding metadata.
        
        Returns:
            A list of unique citation titles. *Does NOT return the response text*
        """
        supports = response.candidates[0].grounding_metadata.grounding_supports
        chunks = response.candidates[0].grounding_metadata.grounding_chunks
        
        list_citations: list[str] = []

        for support in supports:
            if support.grounding_chunk_indices:
                for i in support.grounding_chunk_indices:
                    if i < len(chunks):
                        title = chunks[i].web.title
                        if title not in list_citations:
                            list_citations.append(title)

        return list_citations