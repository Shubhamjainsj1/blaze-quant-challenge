import requests
from dotenv import load_dotenv
load_dotenv()
import os
API_KEY =os.getenv("BLAZEAI_API_KEY")


BLAZE_API_URL = "https://dashboard.withblaze.app/api/graphql-api"


QUERY = """
query WalletContacts($walletAddress: String!){
    walletContacts(walletAddress: $walletAddress){
        twitterHandle
        email
        telegramHandle
    }
}
"""



def fetch_wallet_contacts(wallet_address):

    headers = {
        'Content-Type': 'application/json',
        'x-api-key': API_KEY
    }


    request_body = {"query": QUERY, "variables": {"walletAddress": wallet_address}}

    try:

        response = requests.post(BLAZE_API_URL, json=request_body, headers=headers)
        response.raise_for_status()  # Raise an exception for 4XX and 5XX status codes
        return response.json().get("data", {}).get("walletContacts", {})
    except requests.exceptions.RequestException as e:
        print(f"Error executing GraphQL query: {e}")
        return None





def get_socialmedia_handle(wallet_address):



    wallet_contacts = fetch_wallet_contacts(wallet_address)
    print(wallet_contacts)
    if wallet_contacts:

        twitter_handles = wallet_contacts.get("twitterHandle", [])
        emails = wallet_contacts.get("email", [])
        telegram_handles = wallet_contacts.get("telegramHandle", [])


        social_media_handles = {
            "twitterHandles": twitter_handles,
            "emails": emails,
            "telegramHandles": telegram_handles
        }

        # Print or use the dictionary as needed
        print("Social Media Handles:", social_media_handles)

        return social_media_handles


# v=get_socialmedia_handle("fdassf")