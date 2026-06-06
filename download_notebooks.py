import os
import subprocess
import pandas as pd
import glob
import time

# Configuration
DOMAINS = {
    "Tabular": [
        "titanic", "house-prices-advanced-regression-techniques", "spaceship-titanic",
        "santander-customer-transaction-prediction", "competitive-data-science-predict-future-sales",
        "prudential-life-insurance-assessment", "porto-seguro-safe-driver-prediction",
        "home-credit-default-risk", "bike-sharing-demand", "otto-group-product-classification-challenge",
        "forest-cover-type-prediction", "ghouls-goblins-and-ghosts-boo", "tabular-playground-series-dec-2021"
    ],
    "NLP": [
        "nlp-getting-started", "feedback-prize-english-language-learning", "commonlitreadabilityprize",
        "jigsaw-toxic-comment-classification-challenge", "quora-question-pairs",
        "sentiment-analysis-on-movie-reviews", "word2vec-nlp-tutorial", "bag-of-words-meets-bags-of-popcorn",
        "google-quest-challenge", "natural-language-processing-with-disaster-tweets",
        "jigsaw-unintended-bias-in-toxicity-classification", "tweet-sentiment-extraction"
    ],
    "CV": [
        "digit-recognizer", "dogs-vs-cats-redux-kernels-edition", "cassava-leaf-disease-classification",
        "tgs-salt-identification-challenge", "severstal-steel-defect-detection",
        "state-farm-distracted-driver-detection", "imaterialist-challenge-fashion-2018",
        "painter-by-numbers", "understanding_cloud_organization", "humpback-whale-identification"
    ],
    "Time-Series": [
        "store-sales-time-series-forecasting", "tabular-playground-series-jan-2022",
        "amp-parkinsons-disease-progression-prediction", "g-research-crypto-forecasting",
        "ubiquant-market-prediction", "m5-forecasting-accuracy",
        "walmart-recruiting-store-sales-forecasting", "rossmann-store-sales",
        "web-traffic-time-series-forecasting"
    ],
    "Audio": [
        "birdclef-2023", "birdclef-2022", "freesound-audio-tagging-2019",
        "birdclef-2021", "rfcx-species-audio-detection", "birdsong-recognition",
        "heartbeat-sounds", "birdclef-2024", "esc50-event-classification"
    ],
    "RL": [
        "kore-2022", "santa-2022", "halite", "google-football", "connectx",
        "lux-ai-2021", "hungry-geese"
    ],
    "RecSys": [
        "h-and-m-personalized-fashion-recommendations", "otto-recommender-system",
        "santander-product-recommendation", "elo-merchant-category-recommendation",
        "expedia-hotel-recommendations", "predict-west-nile-virus",
        "instacart-market-basket-analysis", "talkingdata-adtracking-fraud-detection",
        "outbrain-click-prediction", "avazu-ctr-prediction", "recruit-restaurant-visitor-forecasting"
    ],
    "GenAI": [
        "stable-diffusion-image-to-prompts", "llm-prompt-recovery",
        "llm-detect-ai-generated-text", "kaggle-llm-science-exam",
        "llms-you-cant-please-them-all", "drawing-with-llms",
        "feedback-prize-effectiveness", "feedback-prize-2021",
        "google-ai4code", "learning-equality-curriculum-recommendations"
    ],
    "Medical": [
        "rsna-breast-cancer-detection", "histopathologic-cancer-detection",
        "siim-isic-melanoma-classification", "rsna-pneumonia-detection-challenge",
        "osic-pulmonary-fibrosis-progression", "aptos2019-blindness-detection",
        "rsna-intracranial-hemorrhage-detection", "prostate-cancer-grade-assessment",
        "hubmap-kidney-segmentation-connectivity", "sartorius-cell-instance-segmentation",
        "uw-madison-gi-tract-image-segmentation", "pangea-liver-segmentation"
    ]
}

TARGET_PERCENTILES = [0.0, 0.1, 0.2, 0.4, 0.7]

def run_command(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        # print(f"Error running command: {cmd}")
        return None
    return result.stdout

def get_leaderboard(competition):
    print(f"Fetching leaderboard for {competition}...")
    run_command(f"python3 -m kaggle competitions leaderboard -c {competition} --download")
    zip_files = glob.glob(f"{competition}.zip")
    if not zip_files:
        pass
    else:
        run_command(f"unzip -o {competition}.zip")
    
    csv_files = glob.glob(f"*-publicleaderboard-*.csv")
    if not csv_files:
        csv_files = glob.glob(f"{competition}*.csv")
    
    if csv_files:
        csv_files.sort(key=os.path.getmtime, reverse=True)
        try:
            df = pd.read_csv(csv_files[0])
            for f in csv_files:
                if os.path.exists(f): os.remove(f)
            if os.path.exists(f"{competition}.zip"): os.remove(f"{competition}.zip")
            return df
        except Exception:
            return None
    return None

def get_kernels(competition):
    print(f"Fetching kernels for {competition}...")
    output = run_command(f"python3 -m kaggle kernels list --competition {competition} --csv --page-size 500")
    if output:
        from io import StringIO
        try:
            return pd.read_csv(StringIO(output))
        except Exception:
            return None
    return None

def main():
    for domain, competitions in DOMAINS.items():
        print(f"\n=== Processing Domain: {domain} ===")
        domain_dir = domain.replace(" ", "_")
        os.makedirs(domain_dir, exist_ok=True)
        
        for comp in competitions:
            print(f"\nProcessing Competition: {comp}")
            comp_dir = os.path.join(domain_dir, comp)
            os.makedirs(comp_dir, exist_ok=True)
            
            lb_df = get_leaderboard(comp)
            if lb_df is None:
                print(f"Could not get leaderboard for {comp}")
                continue
            
            kernels_df = get_kernels(comp)
            if kernels_df is None or kernels_df.empty:
                print(f"Could not get kernels for {comp}")
                continue
            
            # Map kernels to ranks
            kernels_df['username'] = kernels_df['ref'].apply(lambda x: str(x).split('/')[0])
            
            # Leaderboard might have multiple users per team
            # Expand TeamMemberUserNames
            lb_expanded = lb_df.assign(TeamMemberUserNames=lb_df['TeamMemberUserNames'].str.split(',')).explode('TeamMemberUserNames')
            
            merged = kernels_df.merge(lb_expanded, left_on='username', right_on='TeamMemberUserNames')
            if merged.empty:
                print(f"No kernels found from leaderboard users in top 100 for {comp}. Trying search...")
                # Fallback: just pick some top kernels if we can't match ranks perfectly
                # But the requirement is specific about ranks.
                # Let's try to search specifically for kernels by users at target ranks if we can.
            
            total_teams = len(lb_df)
            target_ranks = []
            for p in TARGET_PERCENTILES:
                rank = max(1, int(p * total_teams))
                target_ranks.append(rank)
            
            print(f"Target ranks: {target_ranks} (Total teams: {total_teams})")
            
            downloaded_count = 0
            for i, target_rank in enumerate(target_ranks):
                percentile_label = f"{int(TARGET_PERCENTILES[i]*100)}pct" if TARGET_PERCENTILES[i] > 0 else "1st"
                
                if not merged.empty:
                    # Find closest rank in merged
                    merged['rank_diff'] = (merged['Rank'] - target_rank).abs()
                    best_match = merged.sort_values('rank_diff').iloc[0]
                    
                    if best_match['rank_diff'] < max(100, total_teams * 0.1): # Increased slack
                        kernel_ref = best_match['ref']
                        actual_rank = best_match['Rank']
                        target_path = os.path.join(comp_dir, f"rank{actual_rank}_{percentile_label}")
                        if os.path.exists(target_path) and os.listdir(target_path):
                            print(f"Target Rank {target_rank} ({percentile_label}): Already downloaded.")
                            downloaded_count += 1
                            continue
                            
                        print(f"Target Rank {target_rank} ({percentile_label}): Found kernel {kernel_ref} at rank {actual_rank}")
                        os.makedirs(target_path, exist_ok=True)
                        run_command(f"python3 -m kaggle kernels pull {kernel_ref} -p \"{target_path}\"")
                        downloaded_count += 1
                        continue

                # If no match in top kernels, try to find ANY kernel from a user near that rank
                print(f"Target Rank {target_rank} ({percentile_label}): No match in top kernels. Searching users...")
                # Get a small window around target rank
                window = lb_df[(lb_df['Rank'] >= target_rank - 5) & (lb_df['Rank'] <= target_rank + 5)]
                found = False
                for _, row in window.iterrows():
                    user = str(row['TeamMemberUserNames']).split(',')[0]
                    actual_rank = row['Rank']
                    target_path = os.path.join(comp_dir, f"rank{actual_rank}_{percentile_label}")
                    if os.path.exists(target_path) and os.listdir(target_path):
                        print(f"  Rank {actual_rank} ({percentile_label}): Already downloaded.")
                        downloaded_count += 1
                        found = True
                        break

                    # Search kernels for this user in this competition
                    user_kernels_out = run_command(f"python3 -m kaggle kernels list --user {user} --competition {comp} --csv")
                    if user_kernels_out:
                        from io import StringIO
                        uk_df = pd.read_csv(StringIO(user_kernels_out))
                        if not uk_df.empty:
                            kernel_ref = uk_df.iloc[0]['ref']
                            actual_rank = row['Rank']
                            print(f"  Found kernel {kernel_ref} for user {user} at rank {actual_rank}")
                            target_path = os.path.join(comp_dir, f"rank{actual_rank}_{percentile_label}")
                            os.makedirs(target_path, exist_ok=True)
                            run_command(f"python3 -m kaggle kernels pull {kernel_ref} -p \"{target_path}\"")
                            downloaded_count += 1
                            found = True
                            break
                if not found:
                    print(f"  Could not find kernel for target rank {target_rank}")

            print(f"Finished {comp}. Downloaded {downloaded_count} notebooks.")

if __name__ == "__main__":
    main()
