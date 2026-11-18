using System.IO;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;

// Receives the stress state from the Python respiration ML over TCP (same PC).
// Python is the SERVER; this script is the CLIENT and keeps reconnecting.
// Messages are one JSON object per line. Attach to any GameObject.
// Start the Python side first (Streamlit page 7 open), then press Play.
[System.Serializable]
public class PhysioMsg
{
    public string type;
    public int window_index;
    public string status;
    public string state;        // "Low" (calm/baseline) or "High" (stress)
    public float confidence;
    public float stress_prob;
    public float bpm;
    public string model;
}

public class PhysiologicalDataReceiver : MonoBehaviour
{
    public string host = "127.0.0.1";
    public int port = 5005;

    [Header("Latest values (read these from other scripts)")]
    public string State = "Low";
    public float StressProb;
    public float Bpm;
    public int WindowIndex;
    public bool HasData;
    public bool Connected;

    Thread thread;
    volatile bool running;
    volatile string latestJson;   // written by thread, consumed in Update()
    TcpClient client;

    void Start()
    {
        running = true;
        thread = new Thread(ConnectLoop) { IsBackground = true };
        thread.Start();
    }

    void ConnectLoop()
    {
        while (running)
        {
            try
            {
                client = new TcpClient();
                client.Connect(host, port);
                Connected = true;
                Debug.Log($"PhysiologicalDataReceiver connected to {host}:{port}");
                using (var reader = new StreamReader(client.GetStream()))
                {
                    string line;
                    while (running && (line = reader.ReadLine()) != null)
                        if (line.Length > 0) latestJson = line;
                }
            }
            catch (SocketException) { /* server not up yet - retry */ }
            catch (IOException) { /* connection dropped - retry */ }
            finally
            {
                Connected = false;
                client?.Close();
            }
            if (running) Thread.Sleep(1000);   // wait before reconnecting
        }
    }

    void Update()
    {
        var json = latestJson;
        if (json == null) return;
        latestJson = null;

        var m = JsonUtility.FromJson<PhysioMsg>(json);
        State = m.state;
        StressProb = m.stress_prob;
        Bpm = m.bpm;
        WindowIndex = m.window_index;
        HasData = true;
        Debug.Log($"Window {m.window_index}: {m.state} (stress {m.stress_prob:P0}, {m.bpm} bpm)");

        // TODO: react here, e.g.
        // DrivingDifficulty.Set(State == "High" ? 1 : 3);
    }

    void OnDestroy()
    {
        running = false;
        client?.Close();
    }
}
