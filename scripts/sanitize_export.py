#!/usr/bin/env python3
"""
Dataset Sanitization & Flattening Pipeline for Label Studio Exports.

This script parses multi-modal Label Studio JSON export files to:
1. Standardize Text NER spans by removing trailing/leading whitespaces and re-calculating indices.
2. Flatten temporal Video Bounding Box sequences into structural frame-by-frame data points.
"""

import os
import json
import logging

# Configure elegant logging for dataset tracing
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def sanitize_text_ner(annotation_result):
    """
    Cleans Named Entity Recognition (NER) annotations by stripping whitespaces
    from labeled substrings and recalculating mathematically precise string offsets.
    """
    if "value" not in annotation_result:
        return annotation_result

    val = annotation_result["value"]
    if "text" in val and "start" in val and "end" in val:
        original_text = val["text"]
        cleaned_text = original_text.strip()

        # If token noise exists (like trailing spaces), recalculate spans
        if original_text != cleaned_text:
            leading_spaces = len(original_text) - len(original_text.lstrip())
            
            # Reposition character indices safely
            new_start = val["start"] + leading_spaces
            new_end = new_start + len(cleaned_text)

            logging.warning(
                f"Token boundary noise sanitized: '{original_text}' -> '{cleaned_text}'"
            )
            val["text"] = cleaned_text
            val["start"] = new_start
            val["end"] = new_end

    return annotation_result


def flatten_video_sequence(annotation_result, task_id):
    """
    Extracts nested, temporal keyframe sequences out of <VideoRectangle> types
    and flattens them into structured tabular-friendly dictionary formats.
    """
    flattened_frames = []
    if "value" not in annotation_result or "sequence" not in annotation_result["value"]:
        return flattened_frames

    val = annotation_result["value"]
    labels = val.get("labels", ["Unlabeled"])
    primary_label = labels[0] if labels else "Unlabeled"
    region_id = annotation_result.get("id", "unknown_region")

    # Unpack frame sequence arrays
    for frame_entry in val["sequence"]:
        flattened_frames.append({
            "task_id": task_id,
            "region_id": region_id,
            "label": primary_label,
            "frame": frame_entry.get("frame"),
            "enabled": frame_entry.get("enabled", True),
            "x_pct": frame_entry.get("x"),
            "y_pct": frame_entry.get("y"),
            "width_pct": frame_entry.get("width"),
            "height_pct": frame_entry.get("height"),
            "rotation_deg": frame_entry.get("rotation", 0),
            "timestamp_sec": frame_entry.get("time")
        })
    
    return flattened_frames


def process_label_studio_export(input_json_path, text_output_path, video_output_path):
    """
    Orchestrates the ingestion, processing, split-routing, and exporting
    of multi-modal data annotation structures.
    """
    if not os.path.exists(input_json_path):
        logging.error(f"Target dataset file not found: {input_json_path}")
        return

    with open(input_json_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    sanitized_tasks = []
    all_flattened_video_rows = []

    logging.info(f"Ingesting {len(tasks)} raw items from Label Studio export mapping...")

    for task in tasks:
        task_id = task.get("id")
        
        # Route through annotation schemas
        if "annotations" in task:
            for annotation in task["annotations"]:
                if "result" in annotation:
                    cleaned_results = []
                    for result in annotation["result"]:
                        # Route Type 1: Text Labels
                        if result.get("type") == "labels":
                            result = sanitize_text_ner(result)
                            cleaned_results.append(result)
                        
                        # Route Type 2: Video Trajectories
                        elif result.get("type") == "videorectangle":
                            video_rows = flatten_video_sequence(result, task_id)
                            all_flattened_video_rows.extend(video_rows)
                            cleaned_results.append(result)
                        else:
                            cleaned_results.append(result)
                    
                    annotation["result"] = cleaned_results
        
        sanitized_tasks.append(task)

    # Save outputs back to target locations
    os.makedirs(os.path.dirname(text_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(video_output_path), exist_ok=True)

    with open(text_output_path, "w", encoding="utf-8") as f:
        json.dump(sanitized_tasks, f, indent=2)
    logging.info(f"Pristine structural Text data exported cleanly to: {text_output_path}")

    if all_flattened_video_rows:
        with open(video_output_path, "w", encoding="utf-8") as f:
            json.dump(all_flattened_video_rows, f, indent=2)
        logging.info(f"Flattened frame-by-frame sequences exported to: {video_output_path}")


if __name__ == "__main__":
    # Mock file architecture definitions for pipeline tracing
    INPUT_FILE = "data/samples_raw/label_studio_raw_export.json"
    TEXT_OUT = "data/samples_exported/sanitized_text_dataset.json"
    VIDEO_OUT = "data/samples_exported/flattened_video_timeline.json"

    print("--- DATASET PREPROCESSING TERMINAL INITIALIZED ---")
    # To run this script locally, place your actual export data at INPUT_FILE and uncomment line below:
    # process_label_studio_export(INPUT_FILE, TEXT_OUT, VIDEO_OUT)
