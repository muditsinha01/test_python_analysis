import os
import json
import random
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from loguru import logger
from prometheus_client import Counter, Histogram, start_http_server, CollectorRegistry

# Constants
DATASET_DIRECTORY = "./Dataset_1"
ANNOTATIONS_FILE = "map.json"
VULNERABILITY_MARKER = "VULNERABLE LINES"

# Prometheus metrics
REGISTRY = CollectorRegistry()
PREP_COUNTER = Counter('data_prep_total', 'Total number of data preparation operations', registry=REGISTRY)
PREP_DURATION = Histogram('data_prep_duration_seconds', 'Duration of data preparation operations', registry=REGISTRY)

# FastAPI app
app = FastAPI()

# Pydantic models
class Annotation(BaseModel):
    char_ranges: List[List[int]]

class EnhancedAnnotation(BaseModel):
    context: List[str]
    char_ranges: List[List[int]]
    is_vulnerable: int = Field(..., ge=0, le=1)

class EnhancedAnnotations(BaseModel):
    annotations: Dict[str, Dict[str, EnhancedAnnotation]]

@PREP_DURATION.time()
def load_json_annotations(filepath: str) -> Dict:
    """Load annotations from a JSON file."""
    logger.info(f"Loading annotations from {filepath}")
    try:
        with open(filepath, 'r') as file:
            annotations = json.load(file)
        PREP_COUNTER.inc()
        return annotations
    except FileNotFoundError:
        logger.error(f"Annotations file not found: {filepath}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in annotations file: {filepath}")
        raise

@PREP_DURATION.time()
def get_context_lines(file_content: List[str], line_number: int, context_range: int = 5) -> List[str]:
    """Extract context lines around a specific line in a file."""
    start = max(0, line_number - context_range - 1)
    end = min(len(file_content), line_number + context_range)
    return [' '.join(line.strip().split()) for line in file_content[start:end]]

@PREP_DURATION.time()
def enhance_annotations_with_negatives(
    annotations: Dict[str, Dict[str, Annotation]],
    dataset_directory: str,
    context_range: int = 5,
    neg_samples_per_positive: int = 1
) -> EnhancedAnnotations:
    """Enhance annotations with context lines and add negative samples."""
    logger.info("Enhancing annotations with negatives")
    enhanced_annotations = {}
    for filename in os.listdir(dataset_directory):
        file_path = os.path.join(dataset_directory, filename)
        try:
            with open(file_path, 'r') as file:
                file_lines = file.readlines()
        except FileNotFoundError:
            logger.warning(f"File not found: {file_path}")
            continue
        except IOError:
            logger.warning(f"Error reading file: {file_path}")
            continue

        file_annotations = annotations.get(filename, {})
        enhanced_annotations[filename] = {}

        for line_num, char_ranges in file_annotations.items():
            line_num_int = int(line_num)
            context = get_context_lines(file_lines, line_num_int, context_range)
            enhanced_annotations[filename][line_num] = EnhancedAnnotation(
                context=context,
                char_ranges=char_ranges.char_ranges,
                is_vulnerable=1
            )

            all_line_nums = set(range(len(file_lines)))
            non_vul_lines = all_line_nums - set(range(max(0, line_num_int - context_range), 
                                                      min(len(file_lines), line_num_int + context_range + 1)))

            for _ in range(neg_samples_per_positive):
                if not non_vul_lines:
                    break
                non_vul_line_num = random.choice(list(non_vul_lines))
                non_vul_context = get_context_lines(file_lines, non_vul_line_num, context_range)
                enhanced_annotations[filename][str(non_vul_line_num)] = EnhancedAnnotation(
                    context=non_vul_context,
                    char_ranges=[],
                    is_vulnerable=0
                )
                non_vul_lines.remove(non_vul_line_num)

    PREP_COUNTER.inc()
    return EnhancedAnnotations(annotations=enhanced_annotations)

@PREP_DURATION.time()
def save_enhanced_annotations(annotations: EnhancedAnnotations, filepath: str) -> None:
    """Save the enhanced annotations to a JSON file."""
    logger.info(f"Saving enhanced annotations to {filepath}")
    try:
        with open(filepath, 'w') as file:
            json.dump(annotations.dict(), file, indent=4)
        PREP_COUNTER.inc()
    except IOError:
        logger.error(f"Error writing enhanced annotations to file: {filepath}")
        raise

# API endpoints
@app.post("/enhance_annotations/", response_model=EnhancedAnnotations)
async def api_enhance_annotations(annotations: Dict[str, Dict[str, Annotation]]):
    try:
        enhanced = enhance_annotations_with_negatives(annotations, DATASET_DIRECTORY)
        return enhanced
    except Exception as e:
        logger.exception("Error in API call to enhance annotations")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/load_annotations/")
async def api_load_annotations():
    try:
        annotations = load_json_annotations(ANNOTATIONS_FILE)
        return annotations
    except Exception as e:
        logger.exception("Error in API call to load annotations")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Start Prometheus metrics server
    start_http_server(8000, registry=REGISTRY)

    # Main script execution
    try:
        annotations = load_json_annotations(ANNOTATIONS_FILE)
        enhanced_annotations = enhance_annotations_with_negatives(annotations, DATASET_DIRECTORY)
        save_enhanced_annotations(enhanced_annotations, 'enhanced_annotations.json')

        # Print a sample of the enhanced annotations for demonstration
        for key, value in list(enhanced_annotations.annotations.items())[:1]:
            print(key, "\n\t", json.dumps(value, indent=4))
    except Exception as e:
        logger.exception(f"An error occurred during data preparation: {str(e)}")

    # Run the FastAPI app
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)