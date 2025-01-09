from datetime import datetime, date
from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import BaseModel, Field

from carrentaltool import search_car_rentals, book_car_rental, update_car_rental, cancel_car_rental
from excursionstool import search_trip_recommendations, book_excursion, update_excursion, cancel_excursion
from flightstool import fetch_user_flight_information, search_flights, update_ticket_to_new_flight, cancel_ticket
from hoteltool import search_hotels, book_hotel, update_hotel, cancel_hotel
from policytool import lookup_policy
from state import State


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            # configuration = config.get("configurable", {})
            # passenger_id = configuration.get("passenger_id", None)
            # state = {**state, "user_info": passenger_id}
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


class CompleteOrEscalate(BaseModel):
    """A tool to mark the current task as completed and/or to escalate control of the dialog to the main assistant,
    who can re-route the dialog based on the user's needs."""

    cancel: bool = True
    reason: str

    class Config:
        json_schema_extra = {
            "example": {
                "cancel": True,
                "reason": "User changed their mind about the current task.",
            },
            "example 2": {
                "cancel": True,
                "reason": "I have fully completed the task.",
            },
            "example 3": {
                "cancel": False,
                "reason": "I need to search the user's emails or calendar for more information.",
            },
        }


class ToFlightBookingAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle flight updates and cancellations."""

    request: str = Field(
        description="Any necessary followup questions the update flight assistant should clarify before proceeding."
    )


class ToBookCarRental(BaseModel):
    """Transfers work to a specialized assistant to handle car rental bookings."""

    location: str = Field(
        description="The location where the user wants to rent a car."
    )
    start_date: str = Field(description="The start date of the car rental.")
    end_date: str = Field(description="The end date of the car rental.")
    request: str = Field(
        description="Any additional information or requests from the user regarding the car rental."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "location": "Basel",
                "start_date": "2023-07-01",
                "end_date": "2023-07-05",
                "request": "I need a compact car with automatic transmission.",
            }
        }


class ToHotelBookingAssistant(BaseModel):
    """Transfer work to a specialized assistant to handle hotel bookings."""

    location: str = Field(
        description="The location where the user wants to book a hotel."
    )
    checkin_date: str = Field(description="The check-in date for the hotel.")
    checkout_date: str = Field(description="The check-out date for the hotel.")
    request: str = Field(
        description="Any additional information or requests from the user regarding the hotel booking."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "location": "Zurich",
                "checkin_date": "2023-08-15",
                "checkout_date": "2023-08-20",
                "request": "I prefer a hotel near the city center with a room that has a view.",
            }
        }


class ToBookExcursion(BaseModel):
    """Transfers work to a specialized assistant to handle trip recommendation and other excursion bookings."""

    location: str = Field(
        description="The location where the user wants to book a recommended trip."
    )
    request: str = Field(
        description="Any additional information or requests from the user regarding the trip recommendation."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "location": "Lucerne",
                "request": "The user is interested in outdoor activities and scenic views.",
            }
        }


# Haiku is faster and cheaper, but less accurate
# llm = ChatAnthropic(model="claude-3-haiku-20240307")
llm = ChatOllama(model="llama3.2:1B", temperature=1)

primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful customer support assistant for Swiss Airlines. "
            "Your primary role is to search for flight information and company policies to answer customer queries. "
            "If a customer requests to update or cancel a flight, book a car rental, book a hotel, or get trip recommendations, "
            "delegate the task to the appropriate specialized assistant by invoking the corresponding tool. You are not able to make these types of changes yourself."
            " Only the specialized assistants are given permission to do this for the user."
            "The user is not aware of the different specialized assistants, so do not mention them; just quietly delegate through function calls. "
            "Provide detailed information to the customer, and always double-check the database before concluding that information is unavailable. "
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            " If a search comes up empty, expand your search before giving up."
            "\n\nCurrent user flight information:\n<Flights>\n{user_info}\n</Flights>"
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)
primary_assistant_tools = [
    TavilySearchResults(max_results=1),
    search_flights,
    lookup_policy,
]
assistant_runnable = primary_assistant_prompt | llm.bind_tools(
    primary_assistant_tools
    + [
        ToFlightBookingAssistant,
        ToBookCarRental,
        ToHotelBookingAssistant,
        ToBookExcursion,
    ]
)

# primary_assistant_prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are a helpful customer support assistant for Swiss Airlines. "
#             " Use the provided tools to search for flights, company policies, and other information to assist the user's queries. "
#             " When searching, be persistent. Expand your query bounds if the first search returns no results. "
#             " If a search comes up empty, expand your search before giving up."
#             "\n\nCurrent user:\n<User>\n{user_info}\n</User>"
#             "\nCurrent time: {time}.",
#         ),
#         ("placeholder", "{messages}"),
#     ]
# ).partial(time=datetime.now)

# part_1_tools = [
#     TavilySearchResults(
#         max_results=1
#     ),
#     fetch_user_flight_information,
#     search_flights,
#     lookup_policy,
#     update_ticket_to_new_flight,
#     cancel_ticket,
#     search_car_rentals,
#     book_car_rental,
#     update_car_rental,
#     cancel_car_rental,
#     search_hotels,
#     book_hotel,
#     update_hotel,
#     cancel_hotel,
#     search_trip_recommendations,
#     book_excursion,
#     update_excursion,
#     cancel_excursion,
# ]

# part_1_assistant_runnable = primary_assistant_prompt | llm.bind_tools(part_1_tools)

# "Read"-only tools (such as retrievers) don't need a user confirmation to use
# part_3_safe_tools = [
#     TavilySearchResults(max_results=1),
#     fetch_user_flight_information,
#     search_flights,
#     lookup_policy,
#     search_car_rentals,
#     search_hotels,
#     search_trip_recommendations,
# ]

# # These tools all change the user's reservations.
# # The user has the right to control what decisions are made
# part_3_sensitive_tools = [
#     update_ticket_to_new_flight,
#     cancel_ticket,
#     book_car_rental,
#     update_car_rental,
#     cancel_car_rental,
#     book_hotel,
#     update_hotel,
#     cancel_hotel,
#     book_excursion,
#     update_excursion,
#     cancel_excursion,
# ]
# sensitive_tool_names = {t.name for t in part_3_sensitive_tools}
# # Our LLM doesn't have to know which nodes it has to route to. In its 'mind', it's just invoking functions.
# part_3_assistant_runnable = primary_assistant_prompt | llm.bind_tools(
#     part_3_safe_tools + part_3_sensitive_tools
# )
