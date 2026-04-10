import operator
from typing import TypedDict, Annotated, List, Optional, Dict, Any


class TravelState(TypedDict):
    user_input:            str
    conversation_history:  List[Dict[str, str]]
    parsed_data:           Optional[Dict[str, Any]]
    parse_error:           Optional[str]
    origin_iata:           Optional[str]
    destination_iata:      Optional[str]
    origin_display:        Optional[str]
    destination_display:   Optional[str]
    airport_resolution_notes: Optional[str]
    validation_result:     Optional[Dict[str, Any]]
    needs_clarification:   bool
    clarification_message: Optional[str]
    flight_results:        Annotated[List[Dict], operator.add]
    total_searches_attempted:  int
    total_searches_succeeded:  int
    best_flight:           Optional[Dict]
    cheapest_flight:       Optional[Dict]
    fastest_flight:        Optional[Dict]
    all_flights_ranked:    Optional[List[Dict]]
    price_insights:        Optional[Dict]
    connecting_flight_results: Optional[List[Dict]]
    alt_transport_results:     Optional[List[Dict]]
    hybrid_itinerary:          Optional[List[Dict]]
    search_mode:               Optional[str]
    final_response:        Optional[str]
    execution_trace:       Annotated[List[Dict], operator.add]


def make_state(user_input: str, history: List = None) -> TravelState:
    return TravelState(
        user_input=user_input,
        conversation_history=history or [],
        parsed_data=None,          parse_error=None,
        origin_iata=None,          destination_iata=None,
        origin_display=None,       destination_display=None,
        airport_resolution_notes=None,
        validation_result=None,    needs_clarification=False,
        clarification_message=None,
        flight_results=[],         total_searches_attempted=0,
        total_searches_succeeded=0,
        best_flight=None,          cheapest_flight=None,
        fastest_flight=None,       all_flights_ranked=None,
        price_insights=None,
        connecting_flight_results=None,
        alt_transport_results=None,
        hybrid_itinerary=None,
        search_mode=None,
        final_response=None,       execution_trace=[],
    )