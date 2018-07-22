"""
Implements a mutation free 4chan thread class.
"""
from functools import reduce


class Thread:
    """
    Represents a thread on 4chan. Contains a linked_list of posts, starting with the OP (original post).
    """

    def __init__(self, title, start_date, op):
        self.title = title
        self.start_date = start_date
        self.op = op
        if type(op) is not Post:
            raise Exception

    def __str__(self):
        return "Thread name: {} \n Date: {} \n posts: \n{}".format(self.title, self.start_date, str(self.op))

    def parse_content(self):
        return "{} \n {}".format(self.op.content, "\n".join(res.parse_content() for res in self.op.responses))


class Post:
    """
    Represents a post in a Thread on 4chan. Contains a set of posts that were sent as replies.
    """

    def __init__(self, title, timestamp, subject, content, resto, responses):
        """
        Creates a Post object.
        """
        self.title = title
        self.timestamp = timestamp
        self.subject = subject
        self.content = content
        self.responses = responses if type(responses) is set else set(responses)
        self.resto = resto

    def add_responses(self, response):
        """
        Adds a new response to the set of responses. Expects responses either to be a single response, a set of responses,
        or a list of responses.
        """
        self.responses = self.responses.union(set(response) if type(response) is not set else response)
        # return Post(self.title, self.timestamp, self.subject, self.content, self.resto,
        #             self.responses.union(set(response) if type(response) is not set else response))

    # Recursively descend the thread and print the results
    def __str__(self):
        return "Post no: {} \n Subject: {} \n content: {} \n resto {} \n{}".format(self.title, self.subject,
                                                                                   self.content, self.resto,
                                                                                   reduce(
                                                                                       lambda x, y: str(x) + '\n' + str(
                                                                                           y),
                                                                                       self.responses, ''))

    def parse_content(self):
        return "{} \n {}".format(self.content, "\n".join(res.parse_content() for res in self.responses))
