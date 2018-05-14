import operator


def print_dissimilarities(user_id, dissimilarities, target_signature_id=None):
    """
    Generate and print a textual representation of a dissimilarity dictionary of some target signature to a user's
    enrolment signatures, sorted by dissimilarity, in ascending order.

    :param user_id:             id of the user for which the dissimilarity has been calculated (e.g. 007)
    :param dissimilarities:     dictionary of type
                                <user enrolment signature id> -> <dissimilarity of target signature to that enrolment signature>
    :param target_signature_id: optional, id / dictionary key / filename of target signature which the user's enrolment
                                signatures were compared to
    :return:                    None
    """
    sorted_dissimilarities = sorted(dissimilarities.items(), key=operator.itemgetter(1))
    print_string = "user: " + user_id
    if target_signature_id is not None:
        print_string += ", target signature id: " + target_signature_id
    for item in sorted_dissimilarities:
        print_string += ", " + item[0] + ": " + str(item[1])
    print(print_string)