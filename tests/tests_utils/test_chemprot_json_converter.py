from io import StringIO
from unittest import TestCase

from preprocessors.chemprot_json_converter import ChemprotJsonConverter


class TestChemprotJsonConverter(TestCase):
    def test_convert(self):
        # Arrange
        sut = ChemprotJsonConverter()
        abstract = "10200320	Cyclopentenone prostaglandins suppress activation of microglia: down-regulation of inducible nitric-oxide synthase by 15-deoxy-Delta12,14-prostaglandin J2.	Mechanisms leading to down-regulation of activated microglia and astrocytes are poorly understood, in spite of the potentially detrimental role of activated glia in neurodegeneration. Prostaglandins, produced both by neurons and glia, may serve as mediators of glial and neuronal functions. We examined the influence of cyclopentenone prostaglandins and their precursors on activated glia. As models of glial activation, production of inducible nitric-oxide synthase (iNOS) was studied in lipopolysaccharide-stimulated rat microglia, a murine microglial cell line BV-2, and IL-1beta-stimulated rat astrocytes. Cyclopentenone prostaglandins were potent inhibitors of iNOS induction and were more effective than their precursors, prostaglandins E2 and D2. 15-Deoxy-Delta12,14-prostaglandin J2 (15d-PGJ2) was the most potent prostaglandin among those tested. In activated microglia, 15d-PGJ2 suppressed iNOS promoter activity, iNOS mRNA, and protein levels. The action of 15d-PGJ2 does not appear to involve its nuclear receptor peroxisome proliferator-activated receptor gamma (PPARgamma) because troglitazone, a specific ligand of PPARgamma, was unable to inhibit iNOS induction, and neither troglitazone nor 15d-PGJ2 could stimulate the activity of a PPAR-dependent promoter in the absence of cotransfected PPARgamma. 15d-PGJ2 did not block nuclear translocation or DNA-binding activity of the transcription factor NFkappaB, but it did inhibit the activity of an NFkappaB reporter construct, suggesting that the mechanism of suppression of microglial iNOS by 15d-PGJ2 may involve interference with NFkappaB transcriptional activity in the nucleus. Thus, our data suggest the existence of a novel pathway mediated by cyclopentenone prostaglandins, which may represent part of a feedback mechanism leading to the cessation of inflammatory glial responses in the brain."
        rel = "10200320	CPR:10	N 	NOT	Arg1:T4	Arg2:T26"
        entities = """10200320	T4	CHEMICAL	1474	1482	15d-PGJ2
10200320	T26	GENE-N	1571	1579	NFkappaB
"""
        expected = [
            {
                'abstract_id': '10200320',
                'abstract': 'Cyclopentenone prostaglandins suppress activation of microglia: down-regulation of inducible nitric-oxide synthase by 15-deoxy-Delta12,14-prostaglandin J2. Mechanisms leading to down-regulation of activated microglia and astrocytes are poorly understood, in spite of the potentially detrimental role of activated glia in neurodegeneration. Prostaglandins, produced both by neurons and glia, may serve as mediators of glial and neuronal functions. We examined the influence of cyclopentenone prostaglandins and their precursors on activated glia. As models of glial activation, production of inducible nitric-oxide synthase (iNOS) was studied in lipopolysaccharide-stimulated rat microglia, a murine microglial cell line BV-2, and IL-1beta-stimulated rat astrocytes. Cyclopentenone prostaglandins were potent inhibitors of iNOS induction and were more effective than their precursors, prostaglandins E2 and D2. 15-Deoxy-Delta12,14-prostaglandin J2 (15d-PGJ2) was the most potent prostaglandin among those tested. In activated microglia, 15d-PGJ2 suppressed iNOS promoter activity, iNOS mRNA, and protein levels. The action of 15d-PGJ2 does not appear to involve its nuclear receptor peroxisome proliferator-activated receptor gamma (PPARgamma) because troglitazone, a specific ligand of PPARgamma, was unable to inhibit iNOS induction, and neither troglitazone nor 15d-PGJ2 could stimulate the activity of a PPAR-dependent promoter in the absence of cotransfected PPARgamma. 15d-PGJ2 did not block nuclear translocation or DNA-binding activity of the transcription factor NFkappaB, but it did inhibit the activity of an NFkappaB reporter construct, suggesting that the mechanism of suppression of microglial iNOS by 15d-PGJ2 may involve interference with NFkappaB transcriptional activity in the nucleus. Thus, our data suggest the existence of a novel pathway mediated by cyclopentenone prostaglandins, which may represent part of a feedback mechanism leading to the cessation of inflammatory glial responses in the brain.',
                'sentence': '15d-PGJ2 did not block nuclear translocation or DNA-binding activity of the transcription factor NFkappaB, but it did inhibit the activity of an NFkappaB reporter construct, suggesting that the mechanism of suppression of microglial iNOS by 15d-PGJ2 may involve interference with NFkappaB transcriptional activity in the nucleus',
                'participant1_id': 'T4',
                'participant1': {
                    'abstract_id': '10200320',
                    'id': 'T4',
                    'entity_type': 'CHEMICAL',
                    'start_pos': 0,
                    'end_pos': 8,
                    'entity_name': '15d-PGJ2'
                },
                'participant2_id': 'T26',
                'participant2': {
                    'abstract_id': '10200320',
                    'id': 'T26',
                    'entity_type': 'GENE-N',
                    'start_pos': 97,
                    'end_pos': 105,
                    'entity_name': 'NFkappaB'
                },
                'relationship_type': 'NOT',
                'relationship_group': 'CPR:10',
                'is_eval': 'N'
            }
        ]

        actual = sut.convert(StringIO(abstract), StringIO(entities), StringIO(rel), None)

        self.assertEqual(expected, actual)
