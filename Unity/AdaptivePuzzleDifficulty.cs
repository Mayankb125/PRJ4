using UnityEngine;
using UnityEngine.Events;

// Adapts puzzle difficulty from the respiration-ML stress state.
//   Stress "High" -> make the puzzle EASIER
//   Stress "Low"  -> make the puzzle HARDER
// Attach next to PhysiologicalDataReceiver (or assign it in the Inspector).
public class AdaptivePuzzleDifficulty : MonoBehaviour
{
    [System.Serializable] public class IntEvent : UnityEvent<int> { }

    public PhysiologicalDataReceiver receiver;

    [Header("Difficulty range")]
    public int minLevel = 1;          // easiest
    public int maxLevel = 5;          // hardest
    public int startLevel = 3;

    [Header("Stability")]
    [Tooltip("Same state must appear this many windows in a row before changing difficulty")]
    public int windowsToConfirm = 2;
    [Tooltip("Ignore window #1: it is only the personal baseline (always Low)")]
    public bool skipBaselineWindow = true;

    [Header("Hook your puzzle here")]
    public IntEvent onDifficultyChanged;   // wire to your puzzle script: SetDifficulty(int)

    public int Level { get; private set; }

    int lastWindow = -1;
    string streakState = "";
    int streak;

    void Start()
    {
        if (receiver == null) receiver = FindObjectOfType<PhysiologicalDataReceiver>();
        Level = Mathf.Clamp(startLevel, minLevel, maxLevel);
        onDifficultyChanged?.Invoke(Level);
    }

    void Update()
    {
        if (receiver == null || !receiver.HasData) return;
        if (receiver.WindowIndex == lastWindow) return;          // act once per new window
        lastWindow = receiver.WindowIndex;

        if (skipBaselineWindow && receiver.WindowIndex <= 1) return;

        streak = (receiver.State == streakState) ? streak + 1 : 1;
        streakState = receiver.State;
        if (streak < windowsToConfirm) return;

        int next = Level;
        if (receiver.State == "High") next = Level - 1;          // stressed -> easier
        else if (receiver.State == "Low") next = Level + 1;      // calm -> harder
        next = Mathf.Clamp(next, minLevel, maxLevel);

        if (next != Level)
        {
            Level = next;
            streak = 0;                                           // require a fresh streak for the next step
            Debug.Log($"Difficulty -> {Level} (state {receiver.State}, stress {receiver.StressProb:P0})");
            onDifficultyChanged?.Invoke(Level);
        }
    }
}
