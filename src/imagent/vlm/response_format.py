from pydantic import BaseModel


class QueryUnderstanding(BaseModel):
    normalized_query: str
    required_details: list[str]
    hypothetical_description: str
