"""
app.py

STAGE 4a: a live web dashboard - a real alternative to reading a scrolling
terminal. This is a small Flask (Python web framework) app.

HOW THIS WORKS, IN PLAIN ENGLISH:
1. A background thread quietly runs a Kafka CONSUMER, reading every new
   message from 'fraud-scored-transactions' (published by ml_fraud_scoring.py)
   the moment it arrives - same idea as our earlier test_consumer.py, just
   running inside a web server instead of printing to a terminal.
2. Each new message gets added to an in-memory list (kept small - only the
   most recent transactions - so the page doesn't grow forever).
3. Your BROWSER connects to a special endpoint ('/stream') using a
   technique called Server-Sent Events (SSE) - the simplest way for a
   server to keep "pushing" new data to an already-open webpage, without
   the browser needing to keep asking "anything new?" over and over.
4. index.html (the actual page) listens for these pushes with plain
   JavaScript and updates the table live, no page refresh needed.

HOW TO RUN:
    Make sure Kafka + your Spark scoring job are already running
    pip install flask
    python dashboard/app.py
Then open http://localhost:5000 in your browser.
"""

import json
import queue
import sys
import threading
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))   # so we can import kafka_config from project root

from flask import Flask, Response, render_template
from kafka import KafkaConsumer

from kafka_config import KAFKA_CONNECTION_KWARGS

SCORED_TOPIC = "fraud-scored-transactions."   # NOTE: real topic name has a trailing dot
MAX_HISTORY = 200   # how many recent transactions to keep in memory

app = Flask(__name__)

# recent_transactions: what a freshly-loaded page shows immediately.
# subscriber_queues: one per currently-open browser tab, so each gets
# every new transaction pushed to it live via SSE.
recent_transactions = deque(maxlen=MAX_HISTORY)
subscriber_queues = []
stats = {"total": 0, "flagged": 0}


def kafka_listener():
    """Runs forever in a background thread, reading from Kafka and fanning
    each new transaction out to every connected browser tab."""
    consumer = KafkaConsumer(
        SCORED_TOPIC,
        **KAFKA_CONNECTION_KWARGS,
        auto_offset_reset="latest",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    )
    print(f"Dashboard listening on Kafka topic '{SCORED_TOPIC}'...")

    for message in consumer:
        txn = message.value
        recent_transactions.append(txn)

        stats["total"] += 1
        if txn["predicted_fraud"] == 1:
            stats["flagged"] += 1

        # Push this transaction to every currently-open browser tab
        for q in subscriber_queues:
            q.put(txn)


@app.route("/")
def index():
    return render_template(
        "index.html",
        initial_transactions=list(reversed(recent_transactions)),
        stats=stats,
    )


@app.route("/stream")
def stream():
    """The SSE endpoint. Each connected browser tab gets its own queue;
    we block here waiting for new items and yield them as they arrive."""
    q = queue.Queue()
    subscriber_queues.append(q)

    def event_stream():
        try:
            while True:
                txn = q.get()   # blocks until a new transaction arrives
                payload = {"txn": txn, "stats": stats}
                yield f"data: {json.dumps(payload)}\n\n"
        finally:
            subscriber_queues.remove(q)

    return Response(event_stream(), mimetype="text/event-stream")


if __name__ == "__main__":
    listener_thread = threading.Thread(target=kafka_listener, daemon=True)
    listener_thread.start()

    print("Dashboard running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True)