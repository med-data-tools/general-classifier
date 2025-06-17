def add_condition(topicId, categoryId, conditionStr):
    found_topic = None
    for topic in topics:
        if topic.get('id') == topicId:
            found_topic = topic
            break

    if found_topic is None:
        print(f"No topic found with ID={topicId}.")
        return

    categories = found_topic.get('categories', [])
    
    for i, cat_tuple in enumerate(categories):
        if len(cat_tuple) == 3:
            (cat_input, cat_box, cat_id) = cat_tuple
            cat_condition = ""  # no condition yet
        else:
            (cat_input, cat_box, cat_id, cat_condition) = cat_tuple

        if cat_id == categoryId:
            new_cat_tuple = (cat_input, cat_box, cat_id, conditionStr)
            categories[i] = new_cat_tuple
            print(f"Condition '{conditionStr}' added to category (ID={categoryId}) in topic (ID={topicId}).")
            return

    print(f"No category (ID={categoryId}) found in topic (ID={topicId}).")
    
    
def remove_condition(topicId, categoryId):
    found_topic = None
    for topic in topics:
        if topic.get('id') == topicId:
            found_topic = topic
            break

    if found_topic is None:
        print(f"No topic found with ID={topicId}.")
        return

    categories = found_topic.get('categories', [])

    for i, cat_tuple in enumerate(categories):
        if len(cat_tuple) == 3:
            (cat_input, cat_box, cat_id) = cat_tuple
            cat_condition = None  
        else:
            (cat_input, cat_box, cat_id, cat_condition) = cat_tuple

        if cat_id == categoryId:
            if len(cat_tuple) == 3:
                print(f"Category (ID={categoryId}) in topic (ID={topicId}) has no condition.")
                return
            else:
                new_cat_tuple = (cat_input, cat_box, cat_id, "")
                categories[i] = new_cat_tuple
                print(f"Condition removed from category (ID={categoryId}) in topic (ID={topicId}).")
                return

    print(f"No category (ID={categoryId}) found in topic (ID={topicId}).")