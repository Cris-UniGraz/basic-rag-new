#!/usr/bin/env python3

import json

# Analyze the log entries for timing discrepancies
log_entries = [
    {
        "total_time": 10.034767866134644,
        "phases": {
            "cache_optimization": 0.9821946620941162,
            "query_generation": 1.6500260829925537,
            "retrieval": 5.408647060394287,
            "processing_reranking": 0.7888860702514648,
            "response_preparation": 0.0015668869018554688,
            "llm_generation": 1.1974828243255615
        }
    },
    {
        "total_time": 67.34684324264526,
        "phases": {
            "cache_optimization": 0.6362159252166748,
            "query_generation": 1.2596378326416016,
            "retrieval": 44.56141376495361,
            "processing_reranking": 1.193289041519165,
            "response_preparation": 0.0014367103576660156,
            "llm_generation": 19.685282468795776
        }
    },
    {
        "total_time": 68.0269525051117,
        "phases": {
            "cache_optimization": 0.8739347457885742,
            "query_generation": 1.6721608638763428,
            "retrieval": 20.245557069778442,
            "processing_reranking": 0.640082836151123,
            "response_preparation": 0.0022745132446289062,
            "llm_generation": 44.58340406417847
        }
    }
]

print("Analysis of timing discrepancies in async pipeline:")
print("=" * 60)

for i, entry in enumerate(log_entries, 1):
    total_time = entry["total_time"]
    phases = entry["phases"]
    
    # Calculate sum of phases
    phases_sum = sum(phases.values())
    
    # Calculate difference
    difference = total_time - phases_sum
    percentage_diff = (difference / total_time) * 100
    
    print(f"\nEntry {i}:")
    print(f"  Total time:     {total_time:.6f} seconds")
    print(f"  Phases sum:     {phases_sum:.6f} seconds")
    print(f"  Difference:     {difference:.6f} seconds ({percentage_diff:.2f}%)")
    print(f"  Phases breakdown:")
    
    for phase_name, phase_time in phases.items():
        print(f"    {phase_name:<20}: {phase_time:.6f}s")

print("\n" + "=" * 60)
print("CONCLUSION:")
print("There are timing discrepancies between total_time and sum of phases.")
print("This suggests there are unmeasured overhead periods between phases.")