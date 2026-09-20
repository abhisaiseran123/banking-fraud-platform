"""
kafka_config.py

Shared connection settings for Redpanda Cloud, used by every script that
talks to Kafka (the producer, the Spark scoring job, and the dashboard).

WHY THIS FILE EXISTS:
Instead of copy-pasting the same bootstrap server address, username, and
password into three different scripts (and risking a typo in one of
them), everything reads from here once. Real credentials live in your
local .env file - NEVER in this file or committed to git.

HOW TO USE:
    from kafka_config import KAFKA_CONNECTION_KWARGS
    producer = KafkaProducer(**KAFKA_CONNECTION_KWARGS, value_serializer=...)
"""

import os
from dotenv import load_dotenv

load_dotenv()   # reads the .env file in your project root into environment variables

REDPANDA_BOOTSTRAP_SERVERS = os.environ["REDPANDA_BOOTSTRAP_SERVERS"]
REDPANDA_USERNAME = os.environ["REDPANDA_USERNAME"]
REDPANDA_PASSWORD = os.environ["REDPANDA_PASSWORD"]

# Common kwargs every KafkaProducer/KafkaConsumer needs to connect securely
# to Redpanda Cloud. SASL_SSL means "encrypted connection, with a
# username/password login" - required for any public, internet-reachable
# Kafka-compatible service, unlike our old local Kafka which had no
# authentication at all (fine for localhost-only, not fine on the internet).
KAFKA_CONNECTION_KWARGS = {
    "bootstrap_servers": REDPANDA_BOOTSTRAP_SERVERS,
    "security_protocol": "SASL_SSL",
    "sasl_mechanism": "SCRAM-SHA-256",
    "sasl_plain_username": REDPANDA_USERNAME,
    "sasl_plain_password": REDPANDA_PASSWORD,
}