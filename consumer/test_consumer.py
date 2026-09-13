"""
test_consumer.py

A throwaway script - NOT part of the final pipeline - used only to prove
that messages are actually flowing through Kafka. It reads from the
'bank-transactions' topic and prints whatever it finds.

In Stage 3, PySpark replaces this script as the "real" consumer.

HOW TO RUN:
    python consumer/test_consumer.py
Then, in a SEPARATE terminal, run the producer:
    python producer/kafka_producer.py
Watch this window - you should see transactions appear here live.

Press Ctrl+C to stop.
"""

import json

from kafka import KafkaConsumer

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "bank-transactions"

consumer = KafkaConsumer(
    KAFKA_TOPIC,
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    auto_offset_reset="latest",   # only show NEW messages from when this starts
    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
)

print(f"Listening on topic '{KAFKA_TOPIC}'... (Ctrl+C to stop)\n")

for message in consumer:
    txn = message.value
    tag = "FRAUD" if txn["is_fraud_actual"] else "  ok  "
    print(f"{tag} | {txn['account_id']} | ${txn['amount']:>9,.2f} | {txn['location']}")