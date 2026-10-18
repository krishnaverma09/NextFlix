import json
import unittest
from unittest.mock import Mock, patch
import requests
from movie_assistant import groq_response

class GroqTest(unittest.TestCase):
    def test_missing_key(self):
        with patch('requests.post') as post:
            self.assertIsNone(groq_response('comedy', ''))
            post.assert_not_called()

    def test_response_and_context(self):
        response = Mock()
        response.json.return_value = {'choices': [{'message': {'content': json.dumps(
            {'reply': 'Try Arrival for thoughtful science fiction.', 'titles': ['Arrival', 'Arrival', 1]})}}]}
        with patch('requests.post', return_value=response) as post:
            result = groq_response('something thoughtful', 'test-key',
                                   [{'role': 'user', 'content': 'I like science fiction'}])
        self.assertEqual(result['titles'], ['Arrival'])
        messages = post.call_args.kwargs['json']['messages']
        self.assertEqual(messages[1]['content'], 'I like science fiction')
        self.assertEqual(messages[-1]['content'], 'something thoughtful')

    def test_rate_limit_and_bad_output(self):
        with patch('requests.post', side_effect=requests.HTTPError):
            self.assertIsNone(groq_response('comedy', 'test-key'))
        for content in ['not json', '{"reply": "", "titles": []}', '{"reply": "Hi", "titles": null}']:
            response = Mock()
            response.json.return_value = {'choices': [{'message': {'content': content}}]}
            with patch('requests.post', return_value=response):
                self.assertIsNone(groq_response('comedy', 'test-key'))
