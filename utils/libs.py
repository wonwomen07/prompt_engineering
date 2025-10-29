import os
from dotenv import load_dotenv
from os import environ
import json
from json import JSONDecodeError


class Libs:

    def load_env(self):
        load_dotenv(".env")