import keyring
import platform

# Set up the API key
api_key = "sk-ragmedia-jdjdlk89j"

# Check the operating system to select the appropriate keyring backend
if "Ubuntu" in platform.platform():
    try:
        # Store the API key securely using the system keyring (Secret Service API or Gnome Keyring)
        keyring.set_password("openai", "api_key", api_key)
        print("API key has been securely stored in the system keychain.")
    except keyring.errors.KeyringError as e:
        print(f"Failed to store the API key in the system keychain: {str(e)}")
else:
    # If running in a headless environment or the system keychain is unavailable, use a file-based backend
    from keyrings.alt.file import PlaintextKeyring
    keyring.set_keyring(PlaintextKeyring())  # Use PlaintextKeyring from keyrings.alt.file
    keyring.set_password("openai", "api_key", api_key)
    print("API key has been stored in a plain text file (for testing only).")

# Verify the key was stored (not required, just for testing)
stored_api_key = keyring.get_password("openai", "api_key")
print(f"Retrieved stored API key: {stored_api_key}")
