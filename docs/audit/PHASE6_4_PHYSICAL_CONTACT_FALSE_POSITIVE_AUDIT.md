# PHASE 6.4 — PHYSICAL CONTACT FALSE POSITIVE AUDIT

## 1. Executive Summary

**Date**: August 2026  
**Artifact Schema**: `phase6_4_physical_contact_fp_audit.json`  
**Purpose**: Diagnose and categorize unmatched verified physical contacts to isolate false positive generation mechanisms.

---

## 2. Physical False Positive Taxonomy Breakdown

| Unmatched Cause Category | Count | Percentage (%) | Mechanism Description |
| :--- | :---: | :---: | :--- |
| **`POST_RALLY_BALL_MOTION`** | 173 | 75.9% | Ball rolling, bouncing slowly, or retrieved by player after point completion. |
| **`DUPLICATE_CONTACT`** | 18 | 7.9% | Multiple candidate peaks triggered around the same true physical contact. |
| **`TIMING_DUPLICATE`** | 21 | 9.2% | Kinematic ripple in adjacent frames outside primary 200 ms matching window. |
| **`PRE_SERVE_RITUAL`** | 16 | 7.0% | Pre-serve ground ball bounces prior to live ball toss. |
| **Total Unmatched Physical Contacts (FP)** | **228** | **100.0%** | All false physical contacts across diagnostic holdout clips. |

---

## 3. Suppression & Debouncing Strategy

1. **Time-Based Debouncing**: $0.35$s minimum physical event interval collapses candidate ripples into a single apex timestamp.
2. **Dead-Ball Kinematic Gating**: Filters low-energy rolling motion ($v_{norm} \le 0.05$ and low acceleration) during post-rally retrieval.
