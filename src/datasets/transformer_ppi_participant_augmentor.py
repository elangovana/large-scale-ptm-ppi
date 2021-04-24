class TransformerPPIParticipantAugmentor:

    def __init__(self, participant1_key: str,
                 participant2_key: str,
                 annotations_dict_key: str,
                 result_key_participant: str,
                 result_key_other: str):
        self.annotations_dict_key = annotations_dict_key
        self.result_key_other = result_key_other
        self.result_key_participant = result_key_participant
        self.participant2_key = participant2_key
        self.participant1_key = participant1_key

    def __call__(self, payload):

        annotations = payload[self.annotations_dict_key]
        participant1_text = payload[self.participant1_key]
        participant2_text = payload[self.participant2_key]

        participants_anno = []
        other_entities_anno = []

        for anno in annotations:
            new_anno = {
                "charOffset": anno["charOffset"]
                , "len": anno["len"]
            }

            if anno["text"] in [participant1_text, participant2_text]:
                participants_anno.append(new_anno)
            else:
                other_entities_anno.append(new_anno)

        payload[self.result_key_participant] = participants_anno
        payload[self.result_key_other] = other_entities_anno

        return payload
