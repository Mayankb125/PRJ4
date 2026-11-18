"""Fake the Python ML side over TCP (no belt needed).

Acts as the TCP SERVER: start this, then press Play in Unity (the receiver connects),
and it sends Low/High readings, one line of JSON each.
Do NOT run it while Streamlit page 7 is open (both use port 5005).
Usage: python Unity/test_unity_sender.py
"""
import json, socket, time
srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.bind(("127.0.0.1", 5005)); srv.listen(1)
print("Waiting for Unity to connect on 127.0.0.1:5005 ... (press Play in Unity)")
conn, addr = srv.accept(); print("Unity connected:", addr)
for i, (state, p) in enumerate([("Low", .1), ("High", .8), ("Low", .2), ("High", .9)], 1):
    msg = {"type": "reading", "window_index": i, "status": "Valid", "state": state,
           "confidence": 0.8, "stress_prob": p, "bpm": 15.0, "model": "test"}
    conn.sendall((json.dumps(msg) + "\n").encode()); print("sent", msg); time.sleep(3)
conn.close(); srv.close()
