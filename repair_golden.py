import csv

FILE = "data/golden/golden_set_final.csv"

labels = {
    "16": ("seat_reservation", "Check the customer's seat reservation and explain that the selected seat was not allocated; provide appropriate seating guidance or investigate the reservation issue.", "yes", "The customer is already at the station and the train is overcrowded, with their reserved seat not allocated, so the current journey requires immediate assistance."),
    "17": ("other", "The message does not contain enough information to determine the customer's request; ask the customer to provide more details.", "yes", "The customer message contains only a link, so there is insufficient information to identify the issue or provide a reliable response."),
    "18": ("seat_reservation", "Check the customer's seat reservation and explain why the requested table and plug seat was not allocated; provide appropriate seating guidance.", "yes", "The customer did not receive the specifically requested seat features and is already describing the failed allocation, so the reservation should be checked."),
    "19": ("refund_compensation", "Explain the applicable refund or compensation process and provide the appropriate claim guidance.", "no", ""),
    "20": ("ticket_change", "Explain whether the ticket can be changed and provide the appropriate ticket amendment process.", "no", ""),
    "21": ("refund_compensation", "Review the mobile ticket failure and refund issue, then direct the customer to the Aftersales team for case-specific assistance.", "yes", "The customer reports that the mobile tickets failed to download and that a refund was not provided, requiring case-specific review of the ticket and refund."),
    "22": ("fare_price", "Review the ticket type and fare charged, explain whether the fare was correct, and investigate the reported ticketing error.", "yes", "The customer disputes a specific ticket sale and unusually high fare and alleges that the wrong ticket was sold, requiring transaction-specific review."),
    "23": ("seat_reservation", "Check the customer's reserved seat and investigate why the window seat was not available, while providing immediate seating assistance for the current journey.", "yes", "The customer reports repeated failure of a reserved window seat and is experiencing the issue during the current journey, requiring staff assistance and investigation."),
    "24": ("complaint_feedback", "Review the customer's compensation case and acknowledge the complaint about the poor journey, while avoiding unnecessary investigation since the customer has already filed for compensation.", "yes", "The customer has already filed for compensation but is also reporting a poor journey experience, so the case should be reviewed in the context of their existing compensation claim and complaint."),
    "25": ("ticket_change", "Review the booking amendment and duplicate tickets, then help the customer resolve the accidental order and explain the applicable refund process.", "yes", "The customer has ended up with duplicate tickets after attempting to amend a booking and cannot complete the refund process, so the booking and refund need case-specific review by Aftersales."),
    "26": ("fare_price", "Explain the applicable fare, price, or booking conditions.", "no", ""),
    "27": ("complaint_feedback", "Acknowledge the concern and provide the appropriate complaint or feedback process.", "no", ""),
    "28": ("complaint_feedback", "Acknowledge the concern and provide the appropriate complaint or feedback process.", "no", ""),
    "29": ("refund_compensation", "Explain the applicable refund or compensation process and provide the appropriate claim guidance.", "no", ""),
    "30": ("ticket_change", "Review the booking amendment and duplicate tickets, then help the customer resolve the accidental order and explain the applicable refund process.", "yes", "The customer has ended up with duplicate tickets after attempting to amend a booking and cannot complete the refund process, so the booking and refund need case-specific review by Aftersales."),
    "31": ("ticket_change", "Explain whether the ticket can be changed and provide the appropriate ticket amendment process.", "no", ""),
    "32": ("seat_reservation", "Check the customer's seat reservation and provide appropriate seating guidance for the current journey.", "yes", "The customer experienced a problem with their seat during the current journey and the circumstances are unclear, so the seating situation should be checked."),
    "33": ("train_status", "Provide relevant information about the disrupted service and acknowledge the alternative travel arrangement.", "no", ""),
    "34": ("ticket_change", "Confirm whether the customer's existing ticket is valid for the proposed alternative route and explain any applicable ticket acceptance arrangements.", "no", ""),
    "35": ("train_status", "Provide the relevant train status, disruption information, and appropriate travel guidance.", "no", ""),
    "36": ("wifi_connectivity", "Provide guidance about the onboard Wi-Fi service and explain the applicable Wi-Fi offering.", "no", ""),
    "37": ("train_status", "Provide immediate information about the delayed service and available travel assistance, while addressing the reported service issues and investigating the disruption.", "yes", "The customer is currently experiencing a significant service disruption involving a delay, missing coaches, poor facilities, and Wi-Fi failure, so the case requires immediate assistance and investigation."),
    "38": ("station_facilities", "Acknowledge the customer's positive feedback about the station lounge.", "no", ""),
    "39": ("other", "Ask the customer to clarify what assistance they need because the message does not identify a specific request.", "yes", "The customer has provided insufficient context to identify the issue or appropriate resolution."),
    "40": ("wifi_connectivity", "Provide guidance for connecting to or troubleshooting the onboard Wi-Fi service.", "no", ""),
    "41": ("wifi_connectivity", "Acknowledge the Wi-Fi connection problem and provide troubleshooting or reset assistance to restore the onboard Wi-Fi service.", "no", ""),
    "42": ("train_status", "Provide current information about the delayed train and available travel assistance, while addressing the significant service disruption.", "yes", "The customer is experiencing a two-hour delay during their journey, requiring current service information and immediate assistance."),
    "43": ("other", "Acknowledge the message because it does not contain a clear support request.", "no", ""),
    "44": ("other", "Ask the customer to provide more information because the message does not contain enough context to identify the request.", "yes", "The message contains only a location/name reference and insufficient information to determine the customer's request."),
    "45": ("ticket_change", "Confirm whether the customer's existing ticket is valid for the proposed alternative route and explain the applicable ticket acceptance or route restrictions.", "no", ""),
    "46": ("other", "Review the customer's previous phone interaction and use the available identifying details to locate the case and provide the appropriate assistance.", "yes", "The customer has no reference number because the interaction was handled by phone, so the support team needs to identify and retrieve the case before resolving the issue."),
    "47": ("refund_compensation", "Review the customer's multiple delayed journeys and determine the appropriate Delay Repay compensation for the affected services.", "yes", "The customer reports multiple delayed journeys and is seeking compensation for several trips, so the individual journeys and compensation eligibility need to be reviewed."),
    "48": ("other", "Ask the customer to provide more information because the reference number alone does not identify the request.", "yes", "The customer has provided only a reference number, so the underlying issue cannot be determined without retrieving the associated case."),
    "49": ("other", "The message is incomplete and does not provide enough context to determine the customer's request; ask the customer to clarify what they need assistance with.", "yes", "The customer message is too incomplete to identify the issue or provide a reliable response without additional context."),
    "50": ("complaint_feedback", "Review the customer's complaint about the poor management of their journey and arrange appropriate follow-up through the complaints process.", "yes", "The customer has explicitly requested contact to discuss a complaint about their journey, so the case requires follow-up by the appropriate support or complaints team."),
    "51": ("other", "Ask the customer to provide clarification because the message is too short to identify the support request.", "yes", "The customer has provided insufficient context to determine the issue or appropriate resolution."),
    "52": ("ticket_change", "Review the customer's request to change the return ticket dates and help complete the amendment after the payment-stage connection failure.", "yes", "The customer was unable to complete a ticket amendment because the connection failed during payment, so the booking needs case-specific assistance from Aftersales."),
    "53": ("refund_compensation", "Acknowledge the customer's positive feedback about the automatic refund for the delayed train.", "no", ""),
    "54": ("ticket_change", "Explain whether the customer's advance-reserved ticket can be used on an earlier train and clarify the applicable travel restrictions.", "no", ""),
    "55": ("refund_compensation", "Acknowledge the customer's positive feedback about the in-app refund process.", "no", ""),
    "56": ("train_status", "Provide immediate information about the cancelled service, available alternative travel arrangements, and applicable compensation guidance.", "yes", "The customer's train has been cancelled due to a crew issue and they are experiencing an active travel disruption, requiring immediate assistance and guidance."),
    "57": ("seat_reservation", "Provide immediate assistance with the lack of available seating and overcrowding on the current service, and check the customer's seating situation.", "yes", "The customer is currently on an overcrowded service with insufficient space and seating, requiring immediate assistance during the journey."),
    "58": ("ticket_change", "Review the customer's ticket purchase and the incorrectly printed tickets, then help resolve the ticketing issue and provide appropriate assistance.", "yes", "The customer reports that the wrong tickets were printed despite proof of purchase and has already had difficulty getting help from staff, so the specific ticketing case requires investigation."),
    "59": ("booking_ticket", "Provide guidance or confirmation regarding the customer's newly booked ticket.", "no", ""),
    "60": ("other", "Acknowledge the message because it does not contain a clear support request.", "no", ""),
    "61": ("train_status", "Provide the relevant train status, disruption information, and appropriate travel guidance.", "no", ""),
    "62": ("other", "Acknowledge the conversational message because it does not contain a support request.", "no", ""),
    "63": ("station_facilities", "Acknowledge the customer's positive feedback about the first-class lounge.", "no", ""),
    "64": ("ticket_change", "Explain whether the customer's ticket is valid after changing services and clarify the applicable ticket requirements when choosing an alternative service.", "no", ""),
    "65": ("wifi_connectivity", "Provide guidance for connecting to or troubleshooting the onboard Wi-Fi service.", "no", "")
}

with open(FILE, "r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

for row in rows:
    if row["id"] in labels:
        intent, resolution, escalate, reason = labels[row["id"]]
        row["intent"] = intent
        row["expected_resolution"] = resolution
        row["should_escalate"] = escalate
        row["escalation_reason"] = reason

fields = [
    "id",
    "customer_message",
    "historical_response",
    "intent",
    "expected_resolution",
    "should_escalate",
    "escalation_reason"
]

with open(FILE, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print("REPAIR COMPLETE")
print("Restored labels: 16-65")
print("Total rows:", len(rows))
print("Labeled rows:", sum(1 for r in rows if r["intent"].strip()))
print("Next unlabeled ID:", next((r["id"] for r in rows if not r["intent"].strip()), "NONE"))
