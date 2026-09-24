#!/bin/bash
# Checks both services are running and the dashboard responds.
# If anything's down, sends a Slack alert (set SLACK_WEBHOOK_URL below).

SLACK_WEBHOOK_URL="PASTE_YOUR_WEBHOOK_URL_HERE"

check_service() {
  if ! systemctl is-active --quiet "$1"; then
    curl -s -X POST -H 'Content-type: application/json' \
      --data "{\"text\":\"🚨 ALERT: $1 is DOWN on the fraud detection server\"}" \
      "$SLACK_WEBHOOK_URL"
  fi
}

check_service fraud-scoring.service
check_service fraud-dashboard.service

if ! curl -s -o /dev/null -w "%{http_code}" http://localhost:5000 | grep -q "200"; then
  curl -s -X POST -H 'Content-type: application/json' \
    --data '{"text":"🚨 ALERT: Dashboard is not responding on port 5000"}' \
    "$SLACK_WEBHOOK_URL"
fi