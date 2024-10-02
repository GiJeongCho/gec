import copy
import json
import re

f = open("src/v1/resources/feedback.json", "r", encoding="utf-8")
feedback_dict = json.load(f)

def make_feedback(tag_type: str, orig_edit_str: str, cor_edit_str: str):

    _feedback_dict = copy.deepcopy(feedback_dict)
    comment = _feedback_dict[tag_type]["comment"]
    comment_word_info = []
    # 설정해놓은 피드백에서 구체적인 틀린 text, 고쳐진 text로 변경합니다.
    if "wrong_word" in comment and "added_word" in comment:
        wrong_start = comment.find("wrong_word")
        add_start = comment.find("added_word")
        
        # 앞에 있는것 부터 교정합니다.
        if wrong_start < add_start:
            wrong_start = comment.find("wrong_word")
            wrong_len = len(orig_edit_str)
            comment = re.sub("wrong_word", orig_edit_str, comment)
            
            add_start = comment.find("added_word")
            add_len = len(cor_edit_str)
            comment = re.sub("added_word", cor_edit_str, comment)
            
            _info = {"word":orig_edit_str, "start_index":wrong_start, "end_index":wrong_start+wrong_len}
            comment_word_info.append(_info)
            _info = {"word":cor_edit_str, "start_index":add_start, "end_index":add_start+add_len}
            comment_word_info.append(_info)
        else:
            add_start = comment.find("added_word")
            add_len = len(cor_edit_str)
            comment = re.sub("added_word", cor_edit_str, comment)
            
            wrong_start = comment.find("wrong_word")
            wrong_len = len(orig_edit_str)
            comment = re.sub("wrong_word", orig_edit_str, comment)
            
            _info = {"word":cor_edit_str, "start_index":add_start, "end_index":add_start+add_len}
            comment_word_info.append(_info)
            _info = {"word":orig_edit_str, "start_index":wrong_start, "end_index":wrong_start+wrong_len}
            comment_word_info.append(_info)
            
    elif "wrong_word" in comment:
        wrong_start = comment.find("wrong_word")
        wrong_len = len(orig_edit_str)
        comment = re.sub("wrong_word", orig_edit_str, comment)
        
        _info = {"word":orig_edit_str, "start_index":wrong_start, "end_index":wrong_start+wrong_len}
        comment_word_info.append(_info)
        
    elif "added_word" in comment:
        add_start = comment.find("added_word")
        add_len = len(cor_edit_str)
        comment = re.sub("added_word", cor_edit_str, comment)
        
        _info = {"word":cor_edit_str, "start_index":add_start, "end_index":add_start+add_len}
        comment_word_info.append(_info)

    _feedback_dict[tag_type]["comment"] = comment
    _feedback_dict[tag_type]["comment_info"] = comment_word_info
    new_order = ["skill_name_en", "skill_name_kr", "comment", "comment_info", "example_1", "example_2"]
    sort_feedback_dict = {key: _feedback_dict[tag_type][key] for key in new_order}
    return sort_feedback_dict