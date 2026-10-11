def rank_candidates(results):
    return sorted(results, key=lambda x: x.get("overall_score", 0), reverse=True)
