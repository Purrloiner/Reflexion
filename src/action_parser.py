import re


def parse_action(response):

    actions = re.findall(
        r"(Search|Lookup|Finish)\[(.*?)\]",
        response
    )

    if len(actions) == 0:
        return None, None

    action, argument = actions[-1]

    return action, argument