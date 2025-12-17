from __future__ import annotations

# --- FastMCP import shim (supports both package layouts) ---
try:
    # Preferred modern package
    from fastmcp import FastMCP
except ImportError:
    # Older installs expose it here
    from mcp.server.fastmcp import FastMCP

import httpx
import logging
import os
import base64
from typing import Optional, Dict, Union, Any, List
from enum import IntEnum, Enum
import re
from pydantic import BaseModel, Field
import dotenv

from datetime import datetime, timedelta
from typing import Literal
import json

# Load environment variables from .env file
dotenv.load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize FastMCP server
mcp = FastMCP("freshdesk-mcp")

FRESHDESK_API_KEY = os.getenv("FRESHDESK_API_KEY")
FRESHDESK_DOMAIN = os.getenv("FRESHDESK_DOMAIN")

HUNGAMA_API_KEY = os.getenv("HUNGAMA_API_KEY")
HUNGAMA_USER_API = os.getenv("HUNGAMA_USER_API")
HUNGAMA_SUBSCRIPTION_API = os.getenv("HUNGAMA_SUBSCRIPTION_API")
HUNGAMA_UNSUBSCRIPTION_API = os.getenv("HUNGAMA_UNSUBSCRIPTION_API")


def parse_link_header(link_header: str) -> Dict[str, Optional[int]]:
    """Parse the Link header to extract pagination information.

    Args:
        link_header: The Link header string from the response

    Returns:
        Dictionary containing next and prev page numbers
    """
    pagination = {
        "next": None,
        "prev": None
    }

    if not link_header:
        return pagination

    # Split multiple links if present
    links = link_header.split(',')

    for link in links:
        # Extract URL and rel
        match = re.search(r'<(.+?)>;\s*rel="(.+?)"', link)
        if match:
            url, rel = match.groups()
            # Extract page number from URL
            page_match = re.search(r'page=(\d+)', url)
            if page_match:
                page_num = int(page_match.group(1))
                pagination[rel] = page_num

    return pagination

# enums of ticket properties
class TicketSource(IntEnum):
    EMAIL = 1
    PORTAL = 2
    PHONE = 3
    CHAT = 7
    FEEDBACK_WIDGET = 9
    OUTBOUND_EMAIL = 10

class TicketStatus(IntEnum):
    OPEN = 2
    PENDING = 3
    RESOLVED = 4
    CLOSED = 5

class TicketPriority(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4

class AgentTicketScope(IntEnum):
    GLOBAL_ACCESS = 1
    GROUP_ACCESS = 2
    RESTRICTED_ACCESS = 3

class UnassignedForOptions(str, Enum):
    THIRTY_MIN = "30m"
    ONE_HOUR = "1h"
    TWO_HOURS = "2h"
    FOUR_HOURS = "4h"
    EIGHT_HOURS = "8h"
    TWELVE_HOURS = "12h"
    ONE_DAY = "1d"
    TWO_DAYS = "2d"
    THREE_DAYS = "3d"

class GroupCreate(BaseModel):
    name: str = Field(..., description="Name of the group")
    description: Optional[str] = Field(None, description="Description of the group")
    agent_ids: Optional[List[int]] = Field(
        default=None,
        description="Array of agent user ids"
    )
    auto_ticket_assign: Optional[int] = Field(
        default=0,
        ge=0,
        le=1,
        description="Automatic ticket assignment type (0 or 1)"
    )
    escalate_to: Optional[int] = Field(
        None,
        description="User ID to whom escalation email is sent if ticket is unassigned"
    )
    unassigned_for: Optional[UnassignedForOptions] = Field(
        default=UnassignedForOptions.THIRTY_MIN,
        description="Time after which escalation email will be sent"
    )

class ContactFieldCreate(BaseModel):
    label: str = Field(..., description="Display name for the field (as seen by agents)")
    label_for_customers: str = Field(..., description="Display name for the field (as seen by customers)")
    type: str = Field(
        ...,
        description="Type of the field",
        pattern="^(custom_text|custom_paragraph|custom_checkbox|custom_number|custom_dropdown|custom_phone_number|custom_url|custom_date)$"
    )
    editable_in_signup: bool = Field(
        default=False,
        description="Set to true if the field can be updated by customers during signup"
    )
    position: int = Field(
        default=1,
        description="Position of the company field"
    )
    required_for_agents: bool = Field(
        default=False,
        description="Set to true if the field is mandatory for agents"
    )
    customers_can_edit: bool = Field(
        default=False,
        description="Set to true if the customer can edit the fields in the customer portal"
    )
    required_for_customers: bool = Field(
        default=False,
        description="Set to true if the field is mandatory in the customer portal"
    )
    displayed_for_customers: bool = Field(
        default=False,
        description="Set to true if the customers can see the field in the customer portal"
    )
    choices: Optional[List[Dict[str, Union[str, int]]]] = Field(
        default=None,
        description="Array of objects in format {'value': 'Choice text', 'position': 1} for dropdown choices"
    )

class CannedResponseCreate(BaseModel):
    title: str = Field(..., description="Title of the canned response")
    content_html: str = Field(..., description="HTML version of the canned response content")
    folder_id: int = Field(..., description="Folder where the canned response gets added")
    visibility: int = Field(
        ...,
        description="Visibility of the canned response (0=all agents, 1=personal, 2=select groups)",
        ge=0,
        le=2
    )
    group_ids: Optional[List[int]] = Field(
        None,
        description="Groups for which the canned response is visible. Required if visibility=2"
    )

@mcp.tool()
async def get_ticket_fields() -> Dict[str, Any]:
    """Get ticket fields from Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/ticket_fields"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}"
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        return response.json()


@mcp.tool()
async def get_tickets(page: Optional[int] = 1, per_page: Optional[int] = 30) -> Dict[str, Any]:
    """Get tickets from Freshdesk with pagination support."""
    # Validate input parameters
    if page < 1:
        return {"error": "Page number must be greater than 0"}

    if per_page < 1 or per_page > 100:
        return {"error": "Page size must be between 1 and 100"}

    url = f"https://{FRESHDESK_DOMAIN}/api/v2/tickets"

    params = {
        "page": page,
        "per_page": per_page
    }

    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()

            # Parse pagination from Link header
            link_header = response.headers.get('Link', '')
            pagination_info = parse_link_header(link_header)

            tickets = response.json()

            return {
                "tickets": tickets,
                "pagination": {
                    "current_page": page,
                    "next_page": pagination_info.get("next"),
                    "prev_page": pagination_info.get("prev"),
                    "per_page": per_page
                }
            }

        except httpx.HTTPStatusError as e:
            return {"error": f"Failed to fetch tickets: {str(e)}"}
        except Exception as e:
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def create_ticket(
    subject: str,
    description: str,
    source: Union[int, str],
    priority: Union[int, str],
    status: Union[int, str],
    email: Optional[str] = None,
    requester_id: Optional[int] = None,
    custom_fields: Optional[Dict[str, Any]] = None,
    additional_fields: Optional[Dict[str, Any]] = None
) -> str:
    """Create a ticket in Freshdesk"""
    # Validate requester information
    if not email and not requester_id:
        return "Error: Either email or requester_id must be provided"

    # Convert string inputs to integers if necessary
    try:
        source_val = int(source)
        priority_val = int(priority)
        status_val = int(status)
    except ValueError:
        return "Error: Invalid value for source, priority, or status"

    # Validate enum values
    if (source_val not in [e.value for e in TicketSource] or
        priority_val not in [e.value for e in TicketPriority] or
        status_val not in [e.value for e in TicketStatus]):
        return "Error: Invalid value for source, priority, or status"

    # Prepare the request data
    data = {
        "subject": subject,
        "description": description,
        "source": source_val,
        "priority": priority_val,
        "status": status_val
    }

    # Add requester information
    if email:
        data["email"] = email
    if requester_id:
        data["requester_id"] = requester_id

    # Add custom fields if provided
    if custom_fields:
        data["custom_fields"] = custom_fields

     # Add any other top-level fields
    if additional_fields:
        data.update(additional_fields)

    url = f"https://{FRESHDESK_DOMAIN}/api/v2/tickets"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=data)
            response.raise_for_status()

            if response.status_code == 201:
                return "Ticket created successfully"

            response_data = response.json()
            return f"Success: {response_data}"

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                # Handle validation errors and check for mandatory custom fields
                error_data = e.response.json()
                if "errors" in error_data:
                    return f"Validation Error: {error_data['errors']}"
            return f"Error: Failed to create ticket - {str(e)}"
        except Exception as e:
            return f"Error: An unexpected error occurred - {str(e)}"

@mcp.tool()
async def update_ticket(ticket_id: int, ticket_fields: Dict[str, Any]) -> Dict[str, Any]:
    """Update a ticket in Freshdesk."""
    if not ticket_fields:
        return {"error": "No fields provided for update"}

    url = f"https://{FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}",
        "Content-Type": "application/json"
    }

    # Separate custom fields from standard fields
    custom_fields = ticket_fields.pop('custom_fields', {})

    # Prepare the update data
    update_data = {}

    # Add standard fields if they are provided
    for field, value in ticket_fields.items():
        update_data[field] = value

    # Add custom fields if they exist
    if custom_fields:
        update_data['custom_fields'] = custom_fields

    async with httpx.AsyncClient() as client:
        try:
            response = await client.put(url, headers=headers, json=update_data)
            response.raise_for_status()

            return {
                "success": True,
                "message": "Ticket updated successfully",
                "ticket": response.json()
            }

        except httpx.HTTPStatusError as e:
            error_message = f"Failed to update ticket: {str(e)}"
            try:
                error_details = e.response.json()
                if "errors" in error_details:
                    error_message = f"Validation errors: {error_details['errors']}"
            except Exception:
                pass
            return {
                "success": False,
                "error": error_message
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"An unexpected error occurred: {str(e)}"
            }

@mcp.tool()
async def delete_ticket(ticket_id: int) -> str:
    """Delete a ticket in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}"
    }
    async with httpx.AsyncClient() as client:
        response = await client.delete(url, headers=headers)
        return response.json()

@mcp.tool()
async def get_ticket(ticket_id: int):
    """Get a ticket in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}"
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        return response.json()

@mcp.tool()
async def search_tickets(query: str) -> Dict[str, Any]:
    """Search for tickets in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/search/tickets"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}"
    }
    params = {"query": query}
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers, params=params)
        return response.json()

@mcp.tool()
async def get_ticket_conversation(ticket_id: int)-> list[Dict[str, Any]]:
    """Get a ticket conversation in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}/conversations"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}"
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        return response.json()

@mcp.tool()
async def create_ticket_reply(ticket_id: int,body: str)-> Dict[str, Any]:
    """Create a reply to a ticket in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}/reply"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}"
    }
    data = {
        "body": body
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=data)
        return response.json()

@mcp.tool()
async def create_ticket_note(ticket_id: int,body: str)-> Dict[str, Any]:
    """Create a note for a ticket in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}/notes"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}"
    }
    data = {
        "body": body
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=data)
        return response.json()

@mcp.tool()
async def update_ticket_conversation(conversation_id: int,body: str)-> Dict[str, Any]:
    """Update a conversation for a ticket in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/conversations/{conversation_id}"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}"
    }
    data = {
        "body": body
    }
    async with httpx.AsyncClient() as client:
        response = await client.put(url, headers=headers, json=data)
        status_code = response.status_code
        if status_code == 200:
            return response.json()
        else:
            return f"Cannot update conversation ${response.json()}"

@mcp.tool()
async def get_agents(page: Optional[int] = 1, per_page: Optional[int] = 30)-> list[Dict[str, Any]]:
    """Get all agents in Freshdesk with pagination support."""
    # Validate input parameters
    if page < 1:
        return {"error": "Page number must be greater than 0"}

    if per_page < 1 or per_page > 100:
        return {"error": "Page size must be between 1 and 100"}
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/agents"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}"
    }
    params = {
        "page": page,
        "per_page": per_page
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers, params=params)
        return response.json()

@mcp.tool()
async def list_contacts(page: Optional[int] = 1, per_page: Optional[int] = 30)-> list[Dict[str, Any]]:
    """List all contacts in Freshdesk with pagination support."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/contacts"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}"
    }
    params = {
        "page": page,
        "per_page": per_page
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers, params=params)
        return response.json()

@mcp.tool()
async def get_contact(contact_id: int)-> Dict[str, Any]:
    """Get a contact in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/contacts/{contact_id}"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}"
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        return response.json()

@mcp.tool()
async def search_contacts(query: str)-> list[Dict[str, Any]]:
    """Search for contacts in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/contacts/autocomplete"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}"
    }
    params = {"term": query}
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers, params=params)
        return response.json()

@mcp.tool()
async def update_contact(contact_id: int, contact_fields: Dict[str, Any])-> Dict[str, Any]:
    """Update a contact in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/contacts/{contact_id}"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}"
    }
    data = {}
    for field, value in contact_fields.items():
        data[field] = value
    async with httpx.AsyncClient() as client:
        response = await client.put(url, headers=headers, json=data)
        return response.json()

@mcp.tool()
async def list_canned_responses(folder_id: int)-> list[Dict[str, Any]]:
    """List all canned responses in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/canned_response_folders/{folder_id}/responses"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}"
    }
    canned_responses = []
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        for canned_response in response.json():
            canned_responses.append(canned_response)
    return canned_responses

@mcp.tool()
async def list_canned_response_folders()-> list[Dict[str, Any]]:
    """List all canned response folders in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/canned_response_folders"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}"
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        return response.json()

@mcp.tool()
async def view_canned_response(canned_response_id: int)-> Dict[str, Any]:
    """View a canned response in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/canned_responses/{canned_response_id}"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}"
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        return response.json()

@mcp.tool()
async def create_canned_response(canned_response_fields: Dict[str, Any])-> Dict[str, Any]:
    """Create a canned response in Freshdesk."""
    # Validate input using Pydantic model
    try:
        validated_fields = CannedResponseCreate(**canned_response_fields)
        # Convert to dict for API request
        canned_response_data = validated_fields.model_dump(exclude_none=True)
    except Exception as e:
        return {"error": f"Validation error: {str(e)}"}

    url = f"https://{FRESHDESK_DOMAIN}/api/v2/canned_responses"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}"
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=canned_response_data)
        return response.json()

@mcp.tool()
async def update_canned_response(canned_response_id: int, canned_response_fields: Dict[str, Any])-> Dict[str, Any]:
    """Update a canned response in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/canned_responses/{canned_response_id}"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}"
    }
    async with httpx.AsyncClient() as client:
        response = await client.put(url, headers=headers, json=canned_response_fields)
        return response.json()

@mcp.tool()
async def create_canned_response_folder(name: str)-> Dict[str, Any]:
    """Create a canned response folder in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/canned_response_folders"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}"
    }
    data = {
        "name": name
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=data)
        return response.json()

@mcp.tool()
async def update_canned_response_folder(folder_id: int, name: str)-> Dict[str, Any]:
    """Update a canned response folder in Freshdesk."""
    print(folder_id, name)
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/canned_response_folders/{folder_id}"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}"
    }
    data = {
        "name": name
    }
    async with httpx.AsyncClient() as client:
        response = await client.put(url, headers=headers, json=data)
        return response.json()

@mcp.tool()
async def list_solution_articles(folder_id: int)-> list[Dict[str, Any]]:
    """List all solution articles in Freshdesk."""
    solution_articles = []
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/solutions/folders/{folder_id}/articles"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}"
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        for article in response.json():
            solution_articles.append(article)
    return solution_articles

@mcp.tool()
async def list_solution_folders(category_id: int)-> list[Dict[str, Any]]:
    """List all solution folders in Freshdesk."""
    if not category_id:
        return {"error": "Category ID is required"}
    
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/solutions/categories/{category_id}/folders"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}"
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        return response.json()

@mcp.tool()
async def list_solution_categories()-> list[Dict[str, Any]]:
    """List all solution categories in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/solutions/categories"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}"
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        return response.json()

@mcp.tool()
async def view_solution_category(category_id: int)-> Dict[str, Any]:
    """View a solution category in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/solutions/categories/{category_id}"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}"
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        return response.json()

@mcp.tool()
async def create_solution_category(category_fields: Dict[str, Any])-> Dict[str, Any]:
    """Create a solution category in Freshdesk."""
    if not category_fields.get("name"):
        return {"error": "Name is required"}

    url = f"https://{FRESHDESK_DOMAIN}/api/v2/solutions/categories"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}"
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=category_fields)
        return response.json()

@mcp.tool()
async def update_solution_category(category_id: int, category_fields: Dict[str, Any])-> Dict[str, Any]:
    """Update a solution category in Freshdesk."""
    if not category_fields.get("name"):
        return {"error": "Name is required"}

    url = f"https://{FRESHDESK_DOMAIN}/api/v2/solutions/categories/{category_id}"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}"
    }
    async with httpx.AsyncClient() as client:
        response = await client.put(url, headers=headers, json=category_fields)
        return response.json()

@mcp.tool()
async def create_solution_category_folder(category_id: int, folder_fields: Dict[str, Any])-> Dict[str, Any]:
    """Create a solution category folder in Freshdesk."""
    if not folder_fields.get("name"):
        return {"error": "Name is required"}
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/solutions/categories/{category_id}/folders"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}"
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=folder_fields)
        return response.json()

@mcp.tool()
async def view_solution_category_folder(folder_id: int)-> Dict[str, Any]:
    """View a solution category folder in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/solutions/folders/{folder_id}"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}"
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        return response.json()

@mcp.tool()
async def update_solution_category_folder(folder_id: int, folder_fields: Dict[str, Any])-> Dict[str, Any]:
    """Update a solution category folder in Freshdesk."""
    if not folder_fields.get("name"):
        return {"error": "Name is required"}
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/solutions/folders/{folder_id}"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}"
    }
    async with httpx.AsyncClient() as client:
        response = await client.put(url, headers=headers, json=folder_fields)
        return response.json()


@mcp.tool()
async def create_solution_article(folder_id: int, article_fields: Dict[str, Any])-> Dict[str, Any]:
    """Create a solution article in Freshdesk."""
    if not article_fields.get("title") or not article_fields.get("status") or not article_fields.get("description"):
        return {"error": "Title, status and description are required"}
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/solutions/folders/{folder_id}/articles"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}"
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=article_fields)
        return response.json()

@mcp.tool()
async def view_solution_article(article_id: int)-> Dict[str, Any]:
    """View a solution article in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/solutions/articles/{article_id}"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}"
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        return response.json()

@mcp.tool()
async def update_solution_article(article_id: int, article_fields: Dict[str, Any])-> Dict[str, Any]:
    """Update a solution article in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/solutions/articles/{article_id}"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}"
    }
    async with httpx.AsyncClient() as client:
        response = await client.put(url, headers=headers, json=article_fields)
        return response.json()

@mcp.tool()
async def view_agent(agent_id: int)-> Dict[str, Any]:
    """View an agent in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/agents/{agent_id}"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}"
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        return response.json()

@mcp.tool()
async def create_agent(agent_fields: Dict[str, Any]) -> Dict[str, Any]:
    """Create an agent in Freshdesk."""
    # Validate mandatory fields
    if not agent_fields.get("email") or not agent_fields.get("ticket_scope"):
        return {
            "error": "Missing mandatory fields. Both 'email' and 'ticket_scope' are required."
        }
    if agent_fields.get("ticket_scope") not in [e.value for e in AgentTicketScope]:
        return {
            "error": "Invalid value for ticket_scope. Must be one of: " + ", ".join([e.name for e in AgentTicketScope])
        }

    url = f"https://{FRESHDESK_DOMAIN}/api/v2/agents"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}"
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=agent_fields)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {
                "error": f"Failed to create agent: {str(e)}",
                "details": e.response.json() if e.response else None
            }

@mcp.tool()
async def update_agent(agent_id: int, agent_fields: Dict[str, Any]) -> Dict[str, Any]:
    """Update an agent in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/agents/{agent_id}"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}"
    }
    async with httpx.AsyncClient() as client:
        response = await client.put(url, headers=headers, json=agent_fields)
        return response.json()

@mcp.tool()
async def search_agents(query: str) -> list[Dict[str, Any]]:
    """Search for agents in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/agents/autocomplete?term={query}"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}"
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        return response.json()

@mcp.tool()
async def list_groups(page: Optional[int] = 1, per_page: Optional[int] = 30)-> list[Dict[str, Any]]:
    """List all groups in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/groups"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}"
    }
    params = {
        "page": page,
        "per_page": per_page
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers, params=params)
        return response.json()

@mcp.tool()
async def create_group(group_fields: Dict[str, Any]) -> Dict[str, Any]:
    """Create a group in Freshdesk."""
    # Validate input using Pydantic model
    try:
        validated_fields = GroupCreate(**group_fields)
        # Convert to dict for API request
        group_data = validated_fields.model_dump(exclude_none=True)
    except Exception as e:
        return {"error": f"Validation error: {str(e)}"}

    url = f"https://{FRESHDESK_DOMAIN}/api/v2/groups"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=group_data)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {
                "error": f"Failed to create group: {str(e)}",
                "details": e.response.json() if e.response else None
            }

@mcp.tool()
async def view_group(group_id: int) -> Dict[str, Any]:
    """View a group in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/groups/{group_id}"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}"
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        return response.json()

@mcp.tool()
async def create_ticket_field(ticket_field_fields: Dict[str, Any]) -> Dict[str, Any]:
    """Create a ticket field in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/admin/ticket_fields"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}"
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=ticket_field_fields)
        return response.json()

@mcp.tool()
async def view_ticket_field(ticket_field_id: int) -> Dict[str, Any]:
    """View a ticket field in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/admin/ticket_fields/{ticket_field_id}"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}"
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        return response.json()

@mcp.tool()
async def update_ticket_field(ticket_field_id: int, ticket_field_fields: Dict[str, Any]) -> Dict[str, Any]:
    """Update a ticket field in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/admin/ticket_fields/{ticket_field_id}"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}"
    }
    async with httpx.AsyncClient() as client:
        response = await client.put(url, headers=headers, json=ticket_field_fields)
        return response.json()

@mcp.tool()
async def update_group(group_id: int, group_fields: Dict[str, Any]) -> Dict[str, Any]:
    """Update a group in Freshdesk."""
    try:
        validated_fields = GroupCreate(**group_fields)
        # Convert to dict for API request
        group_data = validated_fields.model_dump(exclude_none=True)
    except Exception as e:
        return {"error": f"Validation error: {str(e)}"}
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/groups/{group_id}"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}"
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.put(url, headers=headers, json=group_data)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {
                "error": f"Failed to update group: {str(e)}",
                "details": e.response.json() if e.response else None
            }

@mcp.tool()
async def list_contact_fields()-> list[Dict[str, Any]]:
    """List all contact fields in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/contact_fields"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}"
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        return response.json()

@mcp.tool()
async def view_contact_field(contact_field_id: int) -> Dict[str, Any]:
    """View a contact field in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/contact_fields/{contact_field_id}"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}"
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        return response.json()

@mcp.tool()
async def create_contact_field(contact_field_fields: Dict[str, Any]) -> Dict[str, Any]:
    """Create a contact field in Freshdesk."""
    # Validate input using Pydantic model
    try:
        validated_fields = ContactFieldCreate(**contact_field_fields)
        # Convert to dict for API request
        contact_field_data = validated_fields.model_dump(exclude_none=True)
    except Exception as e:
        return {"error": f"Validation error: {str(e)}"}
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/contact_fields"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}"
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=contact_field_data)
        return response.json()

@mcp.tool()
async def update_contact_field(contact_field_id: int, contact_field_fields: Dict[str, Any]) -> Dict[str, Any]:
    """Update a contact field in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/contact_fields/{contact_field_id}"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}"
    }
    async with httpx.AsyncClient() as client:
        response = await client.put(url, headers=headers, json=contact_field_fields)
        return response.json()

@mcp.tool()
async def get_field_properties(field_name: str):
    """Get properties of a specific field by name."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/ticket_fields"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}"
    }
    actual_field_name=field_name
    if field_name == "type":
        actual_field_name="ticket_type"
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()  # Raise error for bad status codes
        fields = response.json()
    # Filter the field by name
    matched_field = next((field for field in fields if field["name"] == actual_field_name), None)

    return matched_field

@mcp.prompt()
def create_ticket_prompt(
    subject: str,
    description: str,
    source: str,
    priority: str,
    status: str,
    email: str
) -> str:
    """Create a ticket in Freshdesk"""
    payload = {
        "subject": subject,
        "description": description,
        "source": source,
        "priority": priority,
        "status": status,
        "email": email,
    }
    return f"""
Kindly create a ticket in Freshdesk using the following payload:

{payload}

If you need to retrieve information about any fields (such as allowed values or internal keys), please use the `get_field_properties()` function.

Notes:
- The "type" field is **not** a custom field; it is a standard system field.
- The "type" field is required but should be passed as a top-level parameter, not within custom_fields.
Make sure to reference the correct keys from `get_field_properties()` when constructing the payload.
"""

@mcp.prompt()
def create_reply_prompt(
    ticket_id:int,
    reply_message: str,
) -> str:
    """Create a reply in Freshdesk"""
    payload = {
        "body":reply_message,
    }
    return f"""
Kindly create a ticket reply in Freshdesk for ticket ID {ticket_id} using the following payload:

{payload}

Notes:
- The "body" field must be in **HTML format** and should be **brief yet contextually complete**.
- When composing the "body", please **review the previous conversation** in the ticket.
- Ensure the tone and style **match the prior replies**, and that the message provides **full context** so the recipient can understand the issue without needing to re-read earlier messages.
"""

@mcp.tool()
async def list_companies(page: Optional[int] = 1, per_page: Optional[int] = 30) -> Dict[str, Any]:
    """List all companies in Freshdesk with pagination support."""
    # Validate input parameters
    if page < 1:
        return {"error": "Page number must be greater than 0"}

    if per_page < 1 or per_page > 100:
        return {"error": "Page size must be between 1 and 100"}

    url = f"https://{FRESHDESK_DOMAIN}/api/v2/companies"

    params = {
        "page": page,
        "per_page": per_page
    }

    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()

            # Parse pagination from Link header
            link_header = response.headers.get('Link', '')
            pagination_info = parse_link_header(link_header)

            companies = response.json()

            return {
                "companies": companies,
                "pagination": {
                    "current_page": page,
                    "next_page": pagination_info.get("next"),
                    "prev_page": pagination_info.get("prev"),
                    "per_page": per_page
                }
            }

        except httpx.HTTPStatusError as e:
            return {"error": f"Failed to fetch companies: {str(e)}"}
        except Exception as e:
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def view_company(company_id: int) -> Dict[str, Any]:
    """Get a company in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/companies/{company_id}"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {"error": f"Failed to fetch company: {str(e)}"}
        except Exception as e:
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def search_companies(query: str) -> Dict[str, Any]:
    """Search for companies in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/companies/autocomplete"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}",
        "Content-Type": "application/json"
    }
    # Use the name parameter as specified in the API
    params = {"name": query}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {"error": f"Failed to search companies: {str(e)}"}
        except Exception as e:
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def find_company_by_name(name: str) -> Dict[str, Any]:
    """Find a company by name in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/companies/autocomplete"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}",
        "Content-Type": "application/json"
    }
    params = {"name": name}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {"error": f"Failed to find company: {str(e)}"}
        except Exception as e:
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def list_company_fields() -> List[Dict[str, Any]]:
    """List all company fields in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/company_fields"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {"error": f"Failed to fetch company fields: {str(e)}"}
        except Exception as e:
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def view_ticket_summary(ticket_id: int) -> Dict[str, Any]:
    """Get the summary of a ticket in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}/summary"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {"error": f"Failed to fetch ticket summary: {str(e)}"}
        except Exception as e:
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def update_ticket_summary(ticket_id: int, body: str) -> Dict[str, Any]:
    """Update the summary of a ticket in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}/summary"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}",
        "Content-Type": "application/json"
    }
    data = {
        "body": body
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.put(url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {"error": f"Failed to update ticket summary: {str(e)}"}
        except Exception as e:
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def delete_ticket_summary(ticket_id: int) -> Dict[str, Any]:
    """Delete the summary of a ticket in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}/summary"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.delete(url, headers=headers)
            if response.status_code == 204:
                return {"success": True, "message": "Ticket summary deleted successfully"}

            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {"error": f"Failed to delete ticket summary: {str(e)}"}
        except Exception as e:
            return {"error": f"An unexpected error occurred: {str(e)}"}

# ============================================================================
# HUNGAMA API INTEGRATION TOOLS
# ============================================================================


@mcp.tool()
async def get_hungama_user_id(mobile_number: str) -> Dict[str, Any]:
    """Get USER_ID (identity) from Hungama User API using mobile number.
    
    Args:
        mobile_number: Mobile number (e.g., "919930130530")
    
    Returns:
        Dict containing USER_ID and user details
    """
    url = HUNGAMA_USER_API
    params = {
        "action": "getuserdetails",
        "username": mobile_number
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params, timeout=10.0)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") == "200":
                return {
                    "success": True,
                    "identity": data["data"]["USER_ID"],
                    "username": data["data"]["USER_NAME"],
                    "data": data["data"]
                }
            else:
                return {
                    "success": False,
                    "error": "User not found",
                    "message": data.get("message", "Unknown error")
                }
                
        except httpx.TimeoutException:
            return {
                "success": False,
                "error": "API timeout - retrying recommended"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get user ID: {str(e)}"
            }

@mcp.tool()
async def get_hungama_subscription(
    identity: str,
    product_id: str = "1",
    device: str = "android",
    variant: str = "v1"
) -> Dict[str, Any]:
    """Get subscription details from Hungama Subscription Status API.
    
    Args:
        identity: USER_ID from get_hungama_user_id
        product_id: Product identifier (default: "1")
        device: Device type (default: "android")
        variant: API variant (default: "v1")
    
    Returns:
        Dict containing subscription details including order_id, plan, status
    """
    url = HUNGAMA_SUBSCRIPTION_API
    params = {"country": "IN"}
    headers = {"Content-Type": "application/json"}
    
    body = {
        "identity": identity,
        "product_id": product_id,
        "device": device,
        "variant": variant
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                url, 
                params=params, 
                headers=headers, 
                json=body,
                timeout=10.0
            )
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("success"):
                # Extract critical data points
                subscriptions = data.get("data", {}).get("subscriptions", {})
                auto_renewal = data.get("data", {}).get("auto_renewal", {})
                
                return {
                    "success": True,
                    "order_id": subscriptions.get("order_id"),
                    "subscription_status": subscriptions.get("subscription_status"),
                    "plan_name": subscriptions.get("plan_name"),
                    "subscription_end_date": subscriptions.get("subscription_end_date"),
                    "days_remaining": subscriptions.get("days_remaining"),
                    "payment_source": subscriptions.get("payment_source"),
                    "plan_price": subscriptions.get("plan_price"),
                    "currency": subscriptions.get("currency"),
                    "auto_renewal_status": auto_renewal.get("status"),
                    "full_data": data
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to get subscription",
                    "message": data.get("message")
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get subscription: {str(e)}"
            }

@mcp.tool()
async def cancel_hungama_subscription(
    order_id: str,
    identity: str,
    platform_id: int = 1,
    product_id: int = 1
) -> Dict[str, Any]:
    """Cancel auto-renewal subscription via Hungama Unsubscription API.
    
    Args:
        order_id: Order ID from subscription details
        identity: USER_ID from user API
        platform_id: Platform identifier (default: 1)
        product_id: Product identifier (default: 1)
    
    Returns:
        Dict containing cancellation status
    """
    url = HUNGAMA_UNSUBSCRIPTION_API
    headers = {
        "Content-Type": "application/json",
        "api-key": HUNGAMA_API_KEY
    }
    
    body = {
        "order_id": order_id,
        "identity": identity,
        "platform_id": platform_id,
        "product_id": product_id
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                url,
                headers=headers,
                json=body,
                timeout=10.0
            )
            response.raise_for_status()
            
            data = response.json()
            
            return {
                "success": data.get("success", False),
                "status_code": data.get("statusCode"),
                "message": data.get("message"),
                "order_id": order_id
            }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to cancel subscription: {str(e)}"
            }

# ============================================================================
# TICKET ENRICHMENT TOOLS
# ============================================================================

@mcp.tool()
async def enrich_ticket_with_subscription_data(ticket_id: int) -> Dict[str, Any]:
    """Fetch user subscription data and add as internal note to ticket.
    
    This tool:
    1. Gets ticket details to extract mobile number
    2. Calls Hungama User API to get identity
    3. Calls Hungama Subscription API to get subscription details
    4. Adds enrichment data as internal note
    
    Args:
        ticket_id: Freshdesk ticket ID
    
    Returns:
        Dict containing enrichment status and data
    """
    # Step 1: Get ticket
    ticket = await get_ticket(ticket_id)
    if "error" in ticket:
        return {"success": False, "error": "Failed to fetch ticket"}
    
    # Step 2: Extract mobile number
    mobile_number = ticket.get("custom_fields", {}).get("cf_mobile_number")
    if not mobile_number:
        # Flag ticket as missing mobile
        await update_ticket(ticket_id, {
            "tags": ticket.get("tags", []) + ["MISSING_MOBILE"]
        })
        return {
            "success": False,
            "error": "Mobile number not found in ticket"
        }
    
    # Step 3: Get USER_ID
    user_data = await get_hungama_user_id(mobile_number)
    if not user_data.get("success"):
        await update_ticket(ticket_id, {
            "tags": ticket.get("tags", []) + ["USER_API_FAILED"]
        })
        return {
            "success": False,
            "error": "Failed to get user ID",
            "details": user_data
        }
    
    identity = user_data.get("identity")
    
    # Step 4: Get subscription details
    subscription_data = await get_hungama_subscription(identity)
    if not subscription_data.get("success"):
        await update_ticket(ticket_id, {
            "tags": ticket.get("tags", []) + ["SUBSCRIPTION_API_FAILED"]
        })
        return {
            "success": False,
            "error": "Failed to get subscription",
            "details": subscription_data
        }
    
    # Step 5: Add enrichment note
    enrichment_note = f"""✅ Enriched by AI
User ID: {identity}
Order ID: {subscription_data.get('order_id')}
Subscription: {subscription_data.get('plan_name')} ({'Active' if subscription_data.get('subscription_status') == 1 else 'Inactive'})
Valid until: {subscription_data.get('subscription_end_date')} ({subscription_data.get('days_remaining')} days)
Payment: {subscription_data.get('payment_source')}
Auto-renewal: {'Enabled' if subscription_data.get('auto_renewal_status') == 1 else 'Disabled'}
Enriched at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
    
    await create_ticket_note(ticket_id, enrichment_note)
    
    return {
        "success": True,
        "ticket_id": ticket_id,
        "mobile_number": mobile_number,
        "identity": identity,
        "enrichment_data": subscription_data
    }

# ============================================================================
# TICKET CLASSIFICATION TOOLS
# ============================================================================

@mcp.tool()
async def classify_ticket(ticket_id: int) -> Dict[str, Any]:
    """Classify ticket into predefined categories using keyword matching.
    
    Categories:
    - AUTO_RENEWAL_CANCELLATION
    - PAYMENT_NOT_ACTIVATED
    - SUBSCRIPTION_NOT_VISIBLE
    - REFUND_REQUEST
    - ARTICLE_FEEDBACK
    - GENERAL_QUERY
    - PROFILE_ISSUES
    - TRANSACTION_STATUS
    - SPAM_PROMOTIONAL
    - NEEDS_MANUAL_REVIEW
    
    Args:
        ticket_id: Freshdesk ticket ID
    
    Returns:
        Dict containing category, confidence score, and tags
    """
    # Get ticket
    ticket = await get_ticket(ticket_id)
    if "error" in ticket:
        return {"success": False, "error": "Failed to fetch ticket"}
    
    # Extract text
    subject = ticket.get("subject", "").lower()
    description = ticket.get("description_text", "").lower()
    ticket_text = f"{subject} {description}"
    
    # Classification keywords (from PRD Section 9.1)
    categories = {
        "AUTO_RENEWAL_CANCELLATION": [
            "stop auto renewal", "cancel auto renewal", "stop e-mandate",
            "cancel e-mandate", "stop subscription", "cancel subscription",
            "unsubscribe", "don't want auto renewal", "stop automatic payment",
            "revoke mandate", "stop recurring charge"
        ],
        "PAYMENT_NOT_ACTIVATED": [
            "payment done but not activated", "paid but no subscription",
            "payment successful but not showing", "subscription not activated",
            "cannot see subscription after payment", "paid yesterday no access"
        ],
        "SUBSCRIPTION_NOT_VISIBLE": [
            "subscription not visible", "cannot see subscription",
            "subscription not showing", "not able to see premium",
            "where is my subscription"
        ],
        "REFUND_REQUEST": [
            "refund", "money back", "return money", "duplicate charge",
            "double charge", "charged twice", "debited twice",
            "unauthorized payment"
        ],
        "ARTICLE_FEEDBACK": [
            "article feedback"
        ],
        "GENERAL_QUERY": [
            "how to", "what is", "general query", "subscription related query",
            "how does it work"
        ],
        "PROFILE_ISSUES": [
            "account hacked", "cannot edit profile", "delete account",
            "delete my account", "delete personal information", "profile locked"
        ],
        "TRANSACTION_STATUS": [
            "transaction status", "payment pending", "payment failed",
            "payment successful", "transaction failed"
        ],
        "SPAM_PROMOTIONAL": [
            "promotional", "event invitation", "marketing", "newsletter"
        ]
    }
    
    # Calculate confidence scores
    scores = {}
    for category, keywords in categories.items():
        matches = sum(1 for keyword in keywords if keyword in ticket_text)
        if len(keywords) > 0:
            scores[category] = (matches / len(keywords)) * 100
        else:
            scores[category] = 0
    
    # Get best category
    if not scores or max(scores.values()) == 0:
        category = "NEEDS_MANUAL_REVIEW"
        confidence = 0
    else:
        category = max(scores, key=scores.get)
        confidence = scores[category]
    
    # If confidence < 80%, flag for manual review
    if confidence < 80 and category != "NEEDS_MANUAL_REVIEW":
        category = "NEEDS_MANUAL_REVIEW"
    
    # Determine priority
    priority_map = {
        "PAYMENT_NOT_ACTIVATED": 3,  # High
        "REFUND_REQUEST": 3,          # High
        "PROFILE_ISSUES": 3,          # High
        "AUTO_RENEWAL_CANCELLATION": 2,  # Medium
        "TRANSACTION_STATUS": 2,      # Medium
        "NEEDS_MANUAL_REVIEW": 2,     # Medium
        "SUBSCRIPTION_NOT_VISIBLE": 2, # Medium
        "ARTICLE_FEEDBACK": 1,        # Low
        "GENERAL_QUERY": 1,           # Low
        "SPAM_PROMOTIONAL": 1         # Low
    }
    priority = priority_map.get(category, 2)
    
    # Update ticket with classification
    tag = f"AI_{category}"
    current_tags = ticket.get("tags", [])
    
    # Remove old AI tags
    current_tags = [t for t in current_tags if not t.startswith("AI_")]
    
    # Add new tag
    current_tags.append(tag)
    
    await update_ticket(ticket_id, {
        "tags": current_tags,
        "priority": priority,
        "custom_fields": {
            **ticket.get("custom_fields", {}),
            "ai_confidence": round(confidence, 2)
        }
    })
    
    # Add classification note
    classification_note = f"""✅ Classified by AI
Category: {category}
Confidence: {round(confidence, 2)}%
Priority: {priority}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
    
    await create_ticket_note(ticket_id, classification_note)
    
    return {
        "success": True,
        "ticket_id": ticket_id,
        "category": category,
        "confidence": round(confidence, 2),
        "priority": priority,
        "tag": tag
    }

# ============================================================================
# RESPONSE TEMPLATE TOOLS
# ============================================================================

# Template storage (you can move this to a JSON file later)
RESPONSE_TEMPLATES = {
    "A1": {
        "name": "Auto-Renewal Cancellation (Active)",
        "condition": "subscription_status == 1",
        "html": """Dear Valued Customer,<br><br>

Thank you for reaching out to Hungama Support.<br><br>

We have successfully cancelled your auto-renewal subscription as requested.<br><br>

<strong>Your Cancellation Details:</strong><br>
✓ Subscription: {{plan_name}}<br>
✓ Order ID: {{order_id}}<br>
✓ Cancellation Date: {{current_date}}<br>
✓ Valid Until: {{subscription_end_date}} ({{days_remaining}} days remaining)<br>
✓ Payment Method: {{payment_source}}<br><br>

<strong>Important Information:</strong><br>
- You will NOT be charged again after {{subscription_end_date}}<br>
- You can continue enjoying Hungama Music until {{subscription_end_date}}<br>
- Your subscription will automatically expire on {{subscription_end_date}}<br>
- All your playlists and favorites will remain saved in your account<br><br>

If you change your mind, you can resubscribe anytime from the Hungama app or website.<br><br>

We're sorry to see you go and hope to serve you again soon!<br><br>

Regards,<br>
<strong>Hungama Support Team</strong><br>
Customer Relations<br>
Hungama Digital Media Entertainment Pvt. Ltd."""
    },
    "A2": {
        "name": "Auto-Renewal Cancellation (Inactive)",
        "condition": "subscription_status == 0",
        "html": """Dear Valued Customer,<br><br>

Thank you for contacting Hungama Support.<br><br>

We have reviewed your account and found that your subscription has already expired.<br><br>

<strong>Your Subscription Details:</strong><br>
- Plan: {{plan_name}}<br>
- Expired On: {{subscription_end_date}}<br>
- Current Status: Inactive<br><br>

<strong>Good News:</strong><br>
Since your subscription has already ended, there will be no further charges to your account. Your auto-renewal has been automatically cancelled upon expiration.<br><br>

If you wish to reactivate your Hungama Music subscription, you can:<br>
1. Open the Hungama app<br>
2. Go to "Subscribe" section<br>
3. Choose your preferred plan<br>
4. Complete the payment<br><br>

We value your patronage and hope to serve you again!<br><br>

Regards,<br>
<strong>Hungama Support Team</strong>"""
    },
    "B1": {
        "name": "Refund Request",
        "html": """Dear Valued Customer,<br><br>

Thank you for bringing this to our attention. We sincerely apologize for any inconvenience caused.<br><br>

We have reviewed your account regarding the duplicate charge issue.<br><br>

<strong>Immediate Action Taken:</strong><br>
✓ Your auto-renewal subscription has been cancelled<br>
✓ Order ID: {{order_id}}<br>
✓ No further charges will be applied<br><br>

<strong>Refund Investigation:</strong><br>
We have initiated an investigation into the duplicate charge reported by you. Our billing team will review your transaction history and process the refund if applicable.<br><br>

<strong>Transaction Details:</strong><br>
- Plan: {{plan_name}}<br>
- Amount: {{currency}} {{plan_price}}<br>
- Payment Method: {{payment_source}}<br><br>

<strong>Next Steps:</strong><br>
- Our billing team will contact you within 24-48 hours<br>
- If refund is applicable, it will be processed within 5-7 business days<br>
- Refund will be credited to your original payment method<br><br>

We deeply regret this error and appreciate your patience while we resolve this matter.<br><br>

Regards,<br>
<strong>Hungama Support Team</strong>"""
    },
    "C1": {
        "name": "Payment Done - Subscription IS Active",
        "html": """Dear Valued Customer,<br><br>

Thank you for contacting Hungama Support.<br><br>

Great news! We have checked your account and your subscription is already active.<br><br>

<strong>Your Active Subscription Details:</strong><br>
✓ Plan: {{plan_name}}<br>
✓ Status: Active<br>
✓ Valid Until: {{subscription_end_date}} ({{days_remaining}} days remaining)<br>
✓ Order ID: {{order_id}}<br>
✓ Payment Method: {{payment_source}}<br><br>

<strong>If you're unable to access premium content, please try:</strong><br><br>

1. <strong>Logout and Login Again:</strong><br>
   • Go to Profile → Logout<br>
   • Close the app completely<br>
   • Reopen and login with your registered mobile: {{mobile_number}}<br><br>

2. <strong>Clear App Cache:</strong><br>
   • Android: Settings → Apps → Hungama → Clear Cache<br>
   • iOS: Delete and reinstall the app<br><br>

3. <strong>Update the App:</strong><br>
   • Ensure you're using the latest version from Play Store/App Store<br><br>

Your subscription is definitely active on our end. These steps should resolve any access issues.<br><br>

Regards,<br>
<strong>Hungama Support Team</strong>"""
    },
    "C2": {
        "name": "Payment Done - Subscription NOT Active",
        "html": """Dear Valued Customer,<br><br>

Thank you for reaching out to Hungama Support.<br><br>

We sincerely apologize for the inconvenience. We have reviewed your account and noticed that your payment has been received but the subscription has not been activated yet.<br><br>

<strong>Your Payment Details:</strong><br>
- Mobile Number: {{mobile_number}}<br>
- Payment Method: {{payment_source}}<br>
- Amount Paid: {{currency}} {{plan_price}}<br><br>

<strong>Immediate Action:</strong><br>
We have escalated this issue to our technical team on high priority. Your subscription will be activated within the next 24 hours.<br><br>

<strong>What Happens Next:</strong><br>
1. Our technical team will manually activate your subscription<br>
2. You will receive a confirmation email once activated<br>
3. You can start enjoying premium content immediately after activation<br><br>

We deeply regret this delay and appreciate your patience.<br><br>

Regards,<br>
<strong>Hungama Support Team</strong>"""
    }
}

# Add this BEFORE the main block (before if __name__ == "__main__":)

@mcp.tool()
async def list_available_templates() -> Dict[str, Any]:
    """List all available response templates.
    
    Returns:
        Dict containing all template IDs, names, and conditions
    """
    templates = []
    for template_id, template_data in RESPONSE_TEMPLATES.items():
        templates.append({
            "id": template_id,
            "name": template_data["name"],
            "condition": template_data.get("condition", "Any")
        })
    
    return {
        "success": True,
        "templates": templates,
        "total": len(templates)
    }

@mcp.tool()
async def get_response_template(
    template_id: str,
    variables: Dict[str, Any]
) -> Dict[str, Any]:
    """Get populated response template with variable substitution.
    
    Args:
        template_id: Template ID (A1, A2, B1, C1, C2, etc.)
        variables: Dict of variables to substitute (e.g., {"plan_name": "MONTHLY PLAN"})
    
    Returns:
        Dict containing populated HTML template
    """
    if template_id not in RESPONSE_TEMPLATES:
        return {
            "success": False,
            "error": f"Template {template_id} not found"
        }
    
    template = RESPONSE_TEMPLATES[template_id]
    html = template["html"]
    
    # Add current_date if not provided
    if "current_date" not in variables:
        variables["current_date"] = datetime.now().strftime("%Y-%m-%d")
    
    # Substitute variables
    for key, value in variables.items():
        placeholder = "{{" + key + "}}"
        html = html.replace(placeholder, str(value))
    
    return {
        "success": True,
        "template_id": template_id,
        "template_name": template["name"],
        "html": html
    }

@mcp.tool()
async def send_templated_response(
    ticket_id: int,
    template_id: str,
    enriched_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Send response to ticket using template with enriched subscription data.
    
    Args:
        ticket_id: Freshdesk ticket ID
        template_id: Template ID (A1, A2, B1, etc.)
        enriched_data: Enrichment data from enrich_ticket_with_subscription_data
    
    Returns:
        Dict containing response status
    """
    # Get template
    template_result = await get_response_template(template_id, enriched_data)
    
    if not template_result.get("success"):
        return template_result
    
    # Send response
    response_result = await create_ticket_reply(
        ticket_id,
        template_result["html"]
    )
    
    # Update ticket status to Resolved
    await update_ticket(ticket_id, {
        "status": 4,  # Resolved
        "tags": ["AI_RESPONDED", "AUTO_CANCELLED"]
    })
    
    # Add processing note
    processing_note = f"""✅ Processed via AI Automation
Template used: {template_id} ({template_result['template_name']})
Response sent: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
    
    await create_ticket_note(ticket_id, processing_note)
    
    return {
        "success": True,
        "ticket_id": ticket_id,
        "template_id": template_id,
        "response": response_result
    }

# ============================================================================
# BATCH PROCESSING TOOLS
# ============================================================================

@mcp.tool()
async def batch_cancel_and_respond(
    ticket_ids: List[int],
    template_id: str = "A1"
) -> Dict[str, Any]:
    """Process multiple tickets: cancel subscriptions and send responses.
    
    For each ticket:
    1. Enrich with subscription data
    2. Cancel subscription via Hungama API
    3. Send templated response
    4. Update ticket status
    
    Args:
        ticket_ids: List of Freshdesk ticket IDs
        template_id: Template to use (default: A1)
    
    Returns:
        Dict containing batch processing results
    """
    results = {
        "successful": [],
        "failed": [],
        "total": len(ticket_ids)
    }
    
    batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    for ticket_id in ticket_ids:
        try:
            # Step 1: Enrich ticket
            enrichment = await enrich_ticket_with_subscription_data(ticket_id)
            
            if not enrichment.get("success"):
                results["failed"].append({
                    "ticket_id": ticket_id,
                    "error": "Enrichment failed",
                    "details": enrichment
                })
                continue
            
            # Step 2: Cancel subscription
            order_id = enrichment["enrichment_data"].get("order_id")
            identity = enrichment["identity"]
            
            if not order_id:
                results["failed"].append({
                    "ticket_id": ticket_id,
                    "error": "No order_id found"
                })
                continue
            
            cancellation = await cancel_hungama_subscription(order_id, identity)
            
            if not cancellation.get("success"):
                results["failed"].append({
                    "ticket_id": ticket_id,
                    "error": "Cancellation API failed",
                    "details": cancellation
                })
                # Flag ticket for manual review
                await update_ticket(ticket_id, {
                    "tags": ["CANCELLATION_FAILED"],
                    "priority": 3
                })
                continue
            
            # Step 3: Send response
            enrichment_data = enrichment["enrichment_data"]
            enrichment_data["mobile_number"] = enrichment["mobile_number"]
            
            response = await send_templated_response(
                ticket_id,
                template_id,
                enrichment_data
            )
            
            if response.get("success"):
                results["successful"].append({
                    "ticket_id": ticket_id,
                    "order_id": order_id
                })
            else:
                results["failed"].append({
                    "ticket_id": ticket_id,
                    "error": "Response sending failed",
                    "details": response
                })
                
        except Exception as e:
            results["failed"].append({
                "ticket_id": ticket_id,
                "error": str(e)
            })
    
    # Create summary
    summary = f"""✅ Batch Processing Complete
Batch ID: {batch_id}
Total tickets: {results['total']}
Successful: {len(results['successful'])}
Failed: {len(results['failed'])}
Processed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
    
    return {
        "success": True,
        "batch_id": batch_id,
        "summary": summary,
        "results": results
    }

# ============================================================================
# DASHBOARD QUERY TOOLS
# ============================================================================

@mcp.tool()
async def get_todays_open_tickets_by_category() -> Dict[str, Any]:
    """Get today's open tickets grouped by AI category.
    
    Returns:
        Dict containing tickets grouped by category with counts
    """
    # Search for today's open tickets
    today = datetime.now().strftime("%Y-%m-%d")
    query = f'status:2 AND created_at:"{today}"'
    
    search_result = await search_tickets(query)
    
    if "error" in search_result:
        return search_result
    
    # Group by category
    categories = {}
    tickets = search_result.get("results", [])
    
    for ticket in tickets:
        tags = ticket.get("tags", [])
        category = "UNCATEGORIZED"
        
        for tag in tags:
            if tag.startswith("AI_"):
                category = tag.replace("AI_", "")
                break
        
        if category not in categories:
            categories[category] = []
        
        categories[category].append({
            "id": ticket["id"],
            "subject": ticket.get("subject"),
            "status": ticket.get("status"),
            "priority": ticket.get("priority"),
            "created_at": ticket.get("created_at")
        })
    
    # Count tickets per category
    counts = {cat: len(tickets) for cat, tickets in categories.items()}
    
    return {
        "success": True,
        "date": today,
        "total_tickets": len(tickets),
        "categories": categories,
        "counts": counts
    }

@mcp.tool()
async def get_yesterdays_closed_tickets() -> Dict[str, Any]:
    """Get yesterday's resolved tickets summary.
    
    Returns:
        Dict containing closed tickets from yesterday
    """
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    query = f'status:4 AND updated_at:"{yesterday}"'
    
    search_result = await search_tickets(query)
    
    if "error" in search_result:
        return search_result
    
    tickets = search_result.get("results", [])
    
    # Count AI vs manual processing
    ai_processed = sum(1 for t in tickets if "AI_RESPONDED" in t.get("tags", []))
    manual_processed = len(tickets) - ai_processed
    
    return {
        "success": True,
        "date": yesterday,
        "total_closed": len(tickets),
        "ai_processed": ai_processed,
        "manual_processed": manual_processed,
        "tickets": tickets
    }

@mcp.tool()
async def get_last_7_days_analysis() -> Dict[str, Any]:
    """Get 7-day ticket analysis with trends.
    
    Returns:
        Dict containing 7-day analysis
    """
    seven_days_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Get all tickets from last 7 days
    query = f'created_at:>"{seven_days_ago}"'
    search_result = await search_tickets(query)
    
    if "error" in search_result:
        return search_result
    
    tickets = search_result.get("results", [])
    
    # Analyze by status
    status_counts = {}
    category_counts = {}
    
    for ticket in tickets:
        status = ticket.get("status")
        status_counts[status] = status_counts.get(status, 0) + 1
        
        tags = ticket.get("tags", [])
        for tag in tags:
            if tag.startswith("AI_"):
                category = tag.replace("AI_", "")
                category_counts[category] = category_counts.get(category, 0) + 1
    
    return {
        "success": True,
        "period": f"{seven_days_ago} to {today}",
        "total_tickets": len(tickets),
        "status_breakdown": status_counts,
        "category_breakdown": category_counts
    }

@mcp.tool()
async def get_urgent_tickets() -> Dict[str, Any]:
    """Get high-priority unresolved tickets requiring attention.
    
    Returns:
        List of urgent tickets
    """
    # Search for high/urgent priority open tickets
    query = 'status:2 AND (priority:3 OR priority:4)'
    
    search_result = await search_tickets(query)
    
    if "error" in search_result:
        return search_result
    
    tickets = search_result.get("results", [])
    
    # Sort by created date (oldest first)
    tickets.sort(key=lambda x: x.get("created_at", ""))
    
    urgent_list = []
    for ticket in tickets:
        urgent_list.append({
            "id": ticket["id"],
            "subject": ticket.get("subject"),
            "priority": ticket.get("priority"),
            "created_at": ticket.get("created_at"),
            "age_hours": (datetime.now() - datetime.fromisoformat(
                ticket.get("created_at", "").replace("Z", "+00:00")
            )).total_seconds() / 3600 if ticket.get("created_at") else 0,
            "tags": ticket.get("tags", [])
        })
    
    return {
        "success": True,
        "total_urgent": len(urgent_list),
        "tickets": urgent_list
    }

@mcp.tool()
async def search_tickets_with_enrichment(
    category: Optional[str] = None,
    status: Optional[int] = None,
    date_range: Optional[str] = None
) -> Dict[str, Any]:
    """Search tickets with enriched subscription data.
    
    Args:
        category: AI category tag (e.g., "AUTO_RENEWAL_CANCELLATION")
        status: Ticket status (2=Open, 3=Pending, 4=Resolved)
        date_range: Date range (e.g., "today", "yesterday", "last_7_days")
    
    Returns:
        Dict containing filtered tickets with enrichment
    """
    # Build query
    query_parts = []
    
    if category:
        query_parts.append(f'tag:"AI_{category}"')
    
    if status:
        query_parts.append(f'status:{status}')
    
    if date_range:
        if date_range == "today":
            date = datetime.now().strftime("%Y-%m-%d")
            query_parts.append(f'created_at:"{date}"')
        elif date_range == "yesterday":
            date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            query_parts.append(f'created_at:"{date}"')
        elif date_range == "last_7_days":
            date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            query_parts.append(f'created_at:>"{date}"')
    
    query = " AND ".join(query_parts) if query_parts else "*"
    
    search_result = await search_tickets(query)
    
    if "error" in search_result:
        return search_result
    
    return {
        "success": True,
        "query": query,
        "tickets": search_result.get("results", []),
        "total": len(search_result.get("results", []))
    }

# ============================================================================
# AUDIT & LOGGING TOOLS
# ============================================================================

@mcp.tool()
async def get_ticket_processing_history(ticket_id: int) -> Dict[str, Any]:
    """Get complete AI processing history for a ticket.
    
    Args:
        ticket_id: Freshdesk ticket ID
    
    Returns:
        Dict containing processing history from internal notes
    """
    # Get ticket conversations to extract internal notes
    conversations = await get_ticket_conversation(ticket_id)
    
    if not isinstance(conversations, list):
        return {"success": False, "error": "Failed to get conversations"}
    
    # Filter for internal notes with AI markers
    ai_notes = []
    for conv in conversations:
        if conv.get("private") and conv.get("body_text"):
            body = conv.get("body_text", "")
            if any(marker in body for marker in ["✅ Classified by AI", "✅ Enriched by AI", "✅ Processed via AI"]):
                ai_notes.append({
                    "timestamp": conv.get("created_at"),
                    "content": body,
                    "user_id": conv.get("user_id")
                })
    
    return {
        "success": True,
        "ticket_id": ticket_id,
        "processing_history": ai_notes,
        "total_ai_actions": len(ai_notes)
    }



# ----------------------------
# MAIN — Streamable HTTP on 127.0.0.1:8001/mcp
# ----------------------------
if __name__ == "__main__":
    print("Starting Freshdesk MCP Streamable HTTP server at http://127.0.0.1:8001/mcp")
    mcp.run(
        "http",               # Streamable HTTP transport
        host="127.0.0.1",
        port=8001,            # Different port from Mixpanel
        path="/mcp"
    )
