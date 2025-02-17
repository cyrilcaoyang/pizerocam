

if __name__ == "__main__":
    client = ImageClient()

    # Please confirm that you have the right server IP address
    client.update_server_ip()

    client.client_session()
