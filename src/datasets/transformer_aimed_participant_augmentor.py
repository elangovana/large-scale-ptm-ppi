class TransformerAimedParticipantAugmentor:

    def __init__(self, participant1_offset_key: str,
                 participant1_len_key: str,
                 participant2_offset_key: str,
                 participant2_len_key: str,
                 result_key):
        self.participant1_len_key = participant1_len_key
        self.participant1_offset_key = participant1_offset_key

        self.participant2_len_key = participant2_len_key
        self.participant2_offset_key = participant2_offset_key

        self.result_column = result_key

    def __call__(self, payload):
        result = [{
            "charOffset": payload[self.participant1_offset_key],
            "len": payload[self.participant1_len_key]

        }, {
            "charOffset": payload[self.participant2_offset_key],
            "len": payload[self.participant2_len_key]
        }
        ]

        payload[self.result_column] = result

        return payload
