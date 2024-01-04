#!/usr/bin/python3
# -*- coding: utf-8 -*

"""Post results to mattermost"""

import requests
import json


def post_message(message,
                 url):
    """Post the 'message' to mattermost"""
    
    payload = {"text": str(message)}
    requests.post(url, data=json.dumps(payload))

    return
