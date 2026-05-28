import gc
import os
from copy import deepcopy

import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image, ImageDraw

from lookwise.method import BaseMethod
from lookwise.qwen2_5_methods import rel_attention_qwen2_5
from lookwise.utils import *


def get_response_with_attention(
    method: BaseMethod,
    annotation,
    ic_examples,
    image_folder: str = None,
    conf_threshold: float = 1.00,
):
    input_image = annotation["input_image"]
    if image_folder is not None:
        input_image = os.path.join(image_folder, input_image)
    question = annotation["question"]
    options = annotation.get("options", None)

    targets = method.generate_visual_cues_using_ic(ic_examples, question)
    targets = [target for target in targets if not include_pronouns(target)]
    answer_type = annotation.get("answer_type", "free_form")
    image_pil = Image.open(input_image).convert("RGB")

    if answer_type != "option_list":
        raise NotImplementedError(f"Unsupported answer_type: {answer_type}")

    initial_images_list = [image_pil]
    answers = []
    first_option_str = options[0]

    question_input_first = format_question(question, first_option_str)
    first_avg_conf, _, _, _ = method.free_form_using_nodes(
        image_pil,
        question_input_first,
        initial_images_list,
        calculate_confidence=True,
    )
    torch.cuda.empty_cache()
    gc.collect()

    entered_crop_branch = False
    if first_avg_conf < conf_threshold:
        print(f"Confidence ({first_avg_conf:.4f}) is below threshold ({conf_threshold}). Triggering zoom-in...")
        images_list, attention_maps, _, _ = process_attention_to_image_list(method, image_pil, targets)
        entered_crop_branch = True
    else:
        print(f"Confidence ({first_avg_conf:.4f}) is sufficient. Using original image.")
        images_list = initial_images_list
        attention_maps = None

    for option_str in options:
        question_input = format_question(question, option_str)
        final_output = method.free_form_using_nodes(
            image_pil,
            question_input,
            images_list,
            calculate_confidence=False,
        )
        answers.append(final_output)
        torch.cuda.empty_cache()
        gc.collect()

    print(answers)
    return answers, images_list, attention_maps, entered_crop_branch


def process_attention_to_image_list(method: BaseMethod, image_pil, targets, include_original=False):
    if not targets:
        return [image_pil], [], [], None

    bbox_size = 224 * 2
    question_tpl_3 = "where is the {}"
    general_question = "Write a general description of the image."
    general_prompt = f"{general_question} Answer the question using a single word or phrase."

    bboxes = []
    attention_maps = []

    for target in targets:
        question = question_tpl_3.format(target)
        prompt = f"{question} Answer the question using a single word or phrase."
        att_map = rel_attention_qwen2_5(
            image_pil,
            prompt,
            general_prompt,
            method.model,
            method.processor,
            target,
        )

        if target.startswith("all "):
            bbox, _ = bbox_from_att_image_nms(
                att_map,
                image_pil.size,
                bbox_size,
                sum_threshold_ratio=0.7,
                nms_iou_threshold=0.2,
            )
        else:
            bbox = bbox_from_att_image_adaptive(att_map, image_pil.size, bbox_size)
        bboxes.append(bbox)

        att = att_map.astype(np.float32)
        att = (att - att.min()) / (att.max() - att.min() + 1e-8)
        cmap = cm.get_cmap("viridis")
        att_color = cmap(att)[:, :, :3]
        att_img = (att_color * 255).astype(np.uint8)

        att_pil_low_res = Image.fromarray(att_img)
        grid_heatmap = att_pil_low_res.resize(image_pil.size, Image.NEAREST)
        attention_maps.append(grid_heatmap)

        smooth_heatmap = att_pil_low_res.resize(image_pil.size, Image.BILINEAR)
        overlay_image = Image.blend(image_pil.convert("RGB"), smooth_heatmap, alpha=0.55)

        draw_overlay = ImageDraw.Draw(overlay_image)
        draw_overlay.rectangle(bbox, outline="red", width=max(1, overlay_image.width // 200))
        attention_maps.append(overlay_image)

    annotated_image = deepcopy(image_pil)
    draw = ImageDraw.Draw(annotated_image)
    for bbox in bboxes:
        draw.rectangle(bbox, outline="red", width=max(1, annotated_image.width // 200))

    images_list = [image_pil if include_original else annotated_image]
    union_bbox = union_all_bboxes(bboxes)
    if union_bbox:
        crop_image = image_pil.crop(union_bbox)
        crop_image = method.resize_image(crop_image)
        images_list.append(crop_image)

    return images_list, attention_maps, bboxes, union_bbox


def format_question(question, option_str):
    return question + "\n" + option_str + "Answer the option letter directly."


def format_question_multichoice(question, options):
    ret = question
    for option in options:
        ret += "\n"
        ret += option
    ret += "\nSelect the best answer to the above multiple-choice question based on the image. Respond with only the letter (A, B, C, D, or E) of the correct option.\nThe best answer is:"
    return ret


def format_question_text_match(question):
    return f"{question} Answer the question using a single word or phrase."


def get_direct_response(
    method: BaseMethod,
    annotation,
    image_folder,
):
    input_image = annotation["input_image"]
    if image_folder is not None:
        input_image = os.path.join(image_folder, input_image)
    question = annotation["question"]
    options = annotation.get("options", None)

    image_pil = Image.open(input_image).convert("RGB")
    image_list = [image_pil]

    answers = []
    for option_str in options:
        question_input = format_question(question, option_str)
        answers.append(method.free_form_using_nodes(image_pil, question_input, image_list))
    return answers
