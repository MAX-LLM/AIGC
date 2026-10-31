import unittest
import json
from unittest.mock import patch

from agent_base import SUPPORT_SIGNPOST, Stage
from mnd_langgraph_agent import (
    LangGraphMNDAgent,
    _declines_background_questions,
    _extract_turn_state,
    _requests_wrap_up,
)


class FakeBackend:
    def __init__(self):
        self.calls = []

    def chat(self, messages):
        self.calls.append(messages)
        prompt = messages[-1]["content"]
        if prompt.startswith("Summarise this MND genetics"):
            return "- Concise retained context"
        if "Stage 2 education explanation" in prompt:
            return "Genetic testing can have positive, negative, or uncertain results."
        if "Wrap up this pre-test" in prompt:
            return "Here is a brief summary and suggested next step."
        if "Verified tool result" in prompt:
            return "SOD1 is associated with some inherited forms of MND."
        if "intended follow-up is:" in prompt:
            intended = prompt.split("intended follow-up is:", 1)[1].split(
                "\n\n", 1
            )[0].strip()
            if "First answer the user's question directly" in prompt:
                return (
                    "A concise answer to the counselling question. "
                    f"{intended}"
                )
            return f"Thank you for explaining that. {intended}"
        return "A concise answer to the counselling question."


class StructuredExtractionBackend(FakeBackend):
    def chat(self, messages):
        prompt = messages[-1]["content"]
        if "Extract structured conversation state" not in prompt:
            return super().chat(messages)

        patient_message = prompt.rsplit("Patient message:", 1)[-1].strip().lower()
        result = {
            "diagnosis_status": "unclear",
            "family_structure_answered": False,
            "family_structure_note": "",
            "family_history_answered": False,
            "family_history_note": "",
            "sharing_preferences_answered": False,
            "sharing_preferences_note": "",
            "result_delivery_answered": False,
            "result_delivery_note": "",
            "testing_context": "pre_test",
            "gene_symbol": "",
            "direct_question": False,
            "unresolved_question": "",
            "concern_note": "",
            "distress_detected": False,
            "skip_current_question": False,
            "skip_remaining_background": False,
            "confidence": 0.98,
        }
        if patient_message in {
            "i haave been diagnosed",
            "diagnosded with mnd",
            "onfirmed diagnosis",
        } or "have been diagnosed" in patient_message:
            result["diagnosis_status"] = "confirmed"
        if "?" in patient_message or patient_message.startswith(
            ("what", "why", "how", "when", "who", "should", "could", "can")
        ):
            result["direct_question"] = True
            result["unresolved_question"] = patient_message[:220]
        if "let'sskip" in patient_message:
            result["skip_remaining_background"] = True
        return json.dumps(result)


class EmptyThenRetryBackend(FakeBackend):
    def chat(self, messages):
        self.calls.append(messages)
        if "Your previous generation was empty" in messages[-1]["content"]:
            return "A complete answer produced by the guarded retry."
        return ""


class WeakExtractionBackend(FakeBackend):
    def chat(self, messages):
        prompt = messages[-1]["content"]
        if "Extract structured conversation state" not in prompt:
            return super().chat(messages)

        return json.dumps(
            {
                "diagnosis_status": "unclear",
                "family_structure_answered": False,
                "family_structure_note": "",
                "family_history_answered": False,
                "family_history_note": "",
                "sharing_preferences_answered": False,
                "sharing_preferences_note": "",
                "result_delivery_answered": False,
                "result_delivery_note": "",
                "testing_context": "pre_test",
                "gene_symbol": "",
                "direct_question": False,
                "unresolved_question": "",
                "concern_note": "",
                "distress_detected": False,
                "skip_current_question": False,
                "skip_remaining_background": False,
                "confidence": 0.4,
            }
        )


class LangGraphAgentTests(unittest.TestCase):
    def setUp(self):
        self.backend = FakeBackend()
        self.agent = LangGraphMNDAgent(self.backend)

    def test_background_information_advances_to_questions(self):
        first = self.agent.ask("I have confirmed MND.")
        self.assertIn("close family", first)
        self.assertIn(SUPPORT_SIGNPOST, first)

        self.agent.ask("I have a wife and two children.")
        self.agent.ask("There is no family history of MND or dementia.")
        self.agent.ask("I would share the result with my family.")
        reply = self.agent.ask("I would receive the result by phone.")

        self.assertEqual(self.agent.state.stage, Stage.QUESTIONS)
        self.assertTrue(self.agent.state.education_given)
        self.assertIn("positive, negative, or uncertain", reply)
        self.assertGreaterEqual(len(self.agent.state.summaries), 1)

    @patch(
        "mnd_langgraph_agent.lookup_gene_info",
        return_value="Local verified SOD1 information.",
    )
    def test_gene_question_uses_tool_node(self, lookup):
        reply = self.agent.ask("What does the SOD1 gene mean?")

        lookup.assert_called_once_with("SOD1")
        self.assertIn("SOD1", reply)
        self.assertEqual(
            self.agent.state.notes["last_gene_lookup"]["gene"],
            "SOD1",
        )

    def test_wrap_up_routes_to_wrap_up_node(self):
        self.agent.state.stage = Stage.QUESTIONS
        reply = self.agent.ask("Please summarise and finish.")

        self.assertEqual(self.agent.state.stage, Stage.WRAP_UP)
        self.assertIn("brief summary", reply)

    def test_result_delivery_written_summary_does_not_trigger_wrap_up(self):
        self.agent.ask("I have confirmed MND.")
        self.agent.ask("I have a partner and two adult children.")
        self.agent.ask("There is no known ALS, MND, or dementia in my family.")
        self.agent.ask(
            "I would tell my partner first and then speak with both children together."
        )
        education = self.agent.ask(
            "I would prefer to receive the result in person with my partner and then have a written summary."
        )

        self.assertEqual(self.agent.state.stage, Stage.QUESTIONS)
        self.assertTrue(self.agent.state.education_given)
        self.assertIn("positive, negative, or uncertain", education)

        reply = self.agent.ask(
            "Can a genetic cause still be found when there is no family history?"
        )

        self.assertEqual(self.agent.state.stage, Stage.QUESTIONS)
        self.assertIn("concise answer", reply)
        self.assertNotIn("brief summary", reply)

    def test_wrap_up_detection_requires_conversation_ending_intent(self):
        self.assertFalse(_requests_wrap_up("I would like a written summary later."))
        self.assertFalse(_requests_wrap_up("Could you summarise genetic testing risks?"))
        self.assertTrue(_requests_wrap_up("Please summarise our conversation."))
        self.assertTrue(_requests_wrap_up("Can we wrap up?"))

    def test_empty_question_generation_is_retried(self):
        agent = LangGraphMNDAgent(EmptyThenRetryBackend())
        agent.state.stage = Stage.QUESTIONS

        reply = agent.ask("What should I ask my genetics team?")

        self.assertIn("guarded retry", reply)
        self.assertTrue(reply.strip())

    def test_pending_diagnosis_counts_as_collected_without_confirmation(self):
        reply = self.agent.ask("I am not yet diagnosed with MND; tests are pending.")

        self.assertTrue(self.agent.state.has_diagnosis_status)
        self.assertFalse(self.agent.state.has_confirmed_diagnosis)
        self.assertEqual(
            self.agent.state.essential_info["diagnosis_status"],
            "pending confirmation",
        )
        self.assertIn("close family", reply)

    def test_natural_negative_family_history_is_recorded(self):
        self.agent.ask("I have confirmed MND.")
        self.agent.ask("I have a partner and one son.")
        reply = self.agent.ask("Nobody in my family has had MND or dementia.")

        self.assertTrue(self.agent.state.has_family_history)
        self.assertIn("who in your family", reply)

    def test_general_question_does_not_set_sharing_preference(self):
        extracted = _extract_turn_state("What can a genetic test tell me?")

        self.assertTrue(extracted.user_is_asking_direct_question)
        self.assertFalse(extracted.has_sharing_preferences)

    def test_declarative_preference_is_not_treated_as_question(self):
        extracted = _extract_turn_state(
            "I would share the test result with my wife and children."
        )

        self.assertTrue(extracted.has_sharing_preferences)
        self.assertFalse(extracted.user_is_asking_direct_question)

    def test_background_direct_question_is_answered_without_intake_prompt(self):
        reply = self.agent.ask("What can a genetic test tell me?")

        self.assertIn("concise answer", reply)
        self.assertNotIn("diagnosed with MND", reply)
        self.assertEqual(self.agent.state.stage, Stage.BACKGROUND)

    def test_questions_stage_does_not_duplicate_latest_user_message(self):
        self.agent.state.stage = Stage.QUESTIONS
        self.agent.ask("What would a negative result mean?")

        messages = self.backend.calls[-1]
        matching_user_messages = [
            message
            for message in messages
            if message["role"] == "user"
            and message["content"] == "What would a negative result mean?"
        ]
        self.assertEqual(len(matching_user_messages), 1)

    def test_questions_stage_is_grounded_with_curated_knowledge(self):
        self.agent.state.stage = Stage.QUESTIONS
        self.agent.ask("What would a negative or uncertain result mean?")

        messages = self.backend.calls[-1]
        grounding_messages = [
            message["content"]
            for message in messages
            if message["role"] == "system"
            and "curated references" in message["content"]
        ]
        self.assertEqual(len(grounding_messages), 1)
        self.assertIn("Post-test genetic counselling", grounding_messages[0])
        self.assertTrue(self.agent.state.notes["last_knowledge_sources"])

    def test_deterministic_direct_question_overrides_weak_llm_extraction(self):
        agent = LangGraphMNDAgent(WeakExtractionBackend())

        reply = agent.ask("Can you explain how ALS can be genetic without family history?")

        self.assertIn("concise answer", reply)
        self.assertNotIn("who in your family", reply)
        extraction = agent.state.notes["last_turn_extraction"]
        self.assertTrue(extraction["deterministic_direct_question"])

    def test_testing_logistics_question_is_routed_to_questions_node(self):
        agent = LangGraphMNDAgent(WeakExtractionBackend())
        reply = agent.ask("Is it a blood test, and how long do results take?")

        self.assertIn("concise answer", reply)
        self.assertNotIn("diagnosed with MND", reply)

    def test_diagnosis_confirmation_uses_previous_question_context(self):
        agent = LangGraphMNDAgent(StructuredExtractionBackend())

        first = agent.ask(
            "I am thinking about genetic testing for MND. What should I know first?"
        )
        self.assertIn("concise answer", first)
        self.assertNotIn("whether you have been diagnosed", first)

        reply = agent.ask(
            "Thanks, I have been diagnosed and have never taken a genetic test before."
        )

        self.assertTrue(agent.state.has_diagnosis_status)
        self.assertTrue(agent.state.has_confirmed_diagnosis)
        self.assertEqual(
            agent.state.essential_info["diagnosis_status"],
            "confirmed",
        )
        self.assertIn("close family", reply)

    def test_bare_yes_confirms_diagnosis_after_diagnosis_question(self):
        self.agent.ask("I want to discuss genetic testing.")
        reply = self.agent.ask("yes")

        self.assertTrue(self.agent.state.has_confirmed_diagnosis)
        self.assertIn("close family", reply)

    def test_no_answers_family_history_question_and_advances(self):
        self.agent.ask("I have confirmed MND.")
        self.agent.ask("I have two children.")
        reply = self.agent.ask("no")

        self.assertTrue(self.agent.state.has_family_history)
        self.assertEqual(
            self.agent.state.essential_info["family_history"],
            "No known family history of MND/ALS or FTD.",
        )
        self.assertIn("who in your family", reply)

    def test_no_plus_direct_question_answers_and_advances_family_structure(self):
        self.agent.ask("I have confirmed MND.")
        reply = self.agent.ask("no can I learn more about the test")

        self.assertTrue(self.agent.state.has_family_structure)
        self.assertEqual(
            self.agent.state.essential_info["family_structure"],
            "No close family members mentioned.",
        )
        self.assertIn("concise answer", reply)
        self.assertNotIn("anyone else in the family", reply)

    def test_full_reported_transcript_does_not_repeat_family_history(self):
        self.agent.ask(
            "I am thinking about genetic testing for MND. What should I know first?"
        )
        self.agent.ask("yes i have been diagnosed")
        self.agent.ask("no can i learn more about test")
        self.agent.ask("2 children")
        reply = self.agent.ask("no")

        self.assertEqual(
            self.agent.state.essential_info["family_structure"],
            "2 children",
        )
        self.assertTrue(self.agent.state.has_family_history)
        self.assertNotIn("Is there anyone else in the family", reply)
        self.assertIn("who in your family", reply)

    def test_short_negative_answers_complete_remaining_background_stages(self):
        self.agent.ask("I have confirmed MND.")
        self.agent.ask("2 children")
        sharing_reply = self.agent.ask("no")
        self.assertIn("who in your family", sharing_reply)

        delivery_reply = self.agent.ask("no")
        self.assertIn("how would you prefer", delivery_reply)

        education_reply = self.agent.ask("no preference")
        self.assertEqual(self.agent.state.stage, Stage.QUESTIONS)
        self.assertTrue(self.agent.state.education_given)
        self.assertIn("positive, negative, or uncertain", education_reply)

    def test_concise_partner_answer_records_sharing_preference(self):
        self.agent.ask("I have confirmed MND.")
        self.agent.ask("I have a partner and two children.")
        self.agent.ask("No one else in my family has MND or dementia.")
        reply = self.agent.ask("my partner")

        self.assertTrue(self.agent.state.has_sharing_preferences)
        self.assertEqual(
            self.agent.state.essential_info["sharing_preferences"],
            "my partner",
        )
        self.assertIn("how would you prefer", reply)

    def test_rephrased_concise_partner_answer_does_not_loop(self):
        self.agent.ask("I have confirmed MND.")
        self.agent.ask("I have two children.")
        self.agent.ask("no")
        reply = self.agent.ask("i said my partner")

        self.assertTrue(self.agent.state.has_sharing_preferences)
        self.assertNotIn("who in your family", reply)
        self.assertIn("how would you prefer", reply)

    def test_misspelled_partner_answer_does_not_loop(self):
        self.agent.ask("I have confirmed MND.")
        self.agent.ask("I have two children.")
        self.agent.ask("no")
        reply = self.agent.ask("my partener. i still dont know if i need to do that")

        self.assertTrue(self.agent.state.has_sharing_preferences)
        self.assertNotIn("who in your family", reply)
        self.assertIn("how would you prefer", reply)

    def test_generated_background_question_keeps_pending_item_in_state(self):
        reply = self.agent.ask("I have confirmed MND.")

        self.assertTrue(reply.startswith("Thank you for explaining that."))
        self.assertEqual(
            self.agent.state.notes["pending_background_item"],
            "family_structure",
        )

    def test_contextual_short_answer_works_with_paraphrased_question(self):
        self.agent.ask("I have confirmed MND.")
        self.agent.history[-1]["content"] = (
            "Who are the important people in your life that you consider close family?"
        )
        reply = self.agent.ask("none")

        self.assertTrue(self.agent.state.has_family_structure)
        self.assertIn("anyone else in the family", reply)

    def test_explicit_skip_defers_remaining_background_questions(self):
        self.agent.ask("yes i have been diagnosed")
        reply = self.agent.ask(
            "i'm not sure if i will take the test yet so we skip this"
        )

        self.assertEqual(self.agent.state.stage, Stage.QUESTIONS)
        self.assertTrue(self.agent.state.education_given)
        self.assertTrue(self.agent.state.notes["background_questions_declined"])
        self.assertIn("family_structure", self.agent.state.notes["deferred_background_items"])
        self.assertNotIn("anyone else in the family", reply.lower())

    def test_uncertainty_alone_does_not_skip_background(self):
        self.assertFalse(
            _declines_background_questions(
                "I am not sure whether I want genetic testing yet."
            )
        )
        self.assertTrue(
            _declines_background_questions(
                "I am not sure whether I want testing yet, so can we skip this?"
            )
        )

    def test_llm_extraction_understands_misspelled_diagnosis(self):
        agent = LangGraphMNDAgent(StructuredExtractionBackend())

        reply = agent.ask("i haave been diagnosed")

        self.assertTrue(agent.state.has_confirmed_diagnosis)
        self.assertEqual(agent.state.essential_info["diagnosis_status"], "confirmed")
        self.assertEqual(agent.state.notes["last_turn_extraction"]["method"], "llm")
        self.assertIn("close family", reply)

    def test_llm_extraction_understands_missing_first_letter(self):
        agent = LangGraphMNDAgent(StructuredExtractionBackend())

        agent.ask("onfirmed diagnosis")

        self.assertTrue(agent.state.has_confirmed_diagnosis)
        self.assertEqual(agent.state.notes["last_turn_extraction"]["method"], "llm")

    def test_llm_extraction_handles_unspaced_skip_typo(self):
        agent = LangGraphMNDAgent(StructuredExtractionBackend())
        agent.ask("diagnosded with mnd")

        reply = agent.ask("i haven't decided yet so let'sskip")

        self.assertEqual(agent.state.stage, Stage.QUESTIONS)
        self.assertTrue(agent.state.education_given)
        self.assertTrue(agent.state.notes["background_questions_declined"])
        self.assertEqual(agent.state.notes["last_turn_extraction"]["method"], "llm")
        self.assertNotIn("family with MND", reply)


if __name__ == "__main__":
    unittest.main()
