rpi_client.py
===

start recording
```
mosquitto_pub -h localhost -p 1883 -t /ict4awe_demo/cmd/record -m '{"action": "start", "id": 1, "event": "door"}'
```

stop recording
```
mosquitto_pub -h localhost -p 1883  -t /ict4awe_demo/cmd/record -m '{"action": "stop", "id": 2}'
```

 create a classifier from recorded events
```
mosquitto_pub -h localhost -p 1883 -t /ict4awe_demo/cmd/classification -m '{"action": "create", "id": 1, "events": ["flush", "shower"]}'
```
