import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recommender import (load_catalog, content_model, content_recommendations,
                         collaborative_model, collaborative_recommendations)
from streamlit.testing.v1 import AppTest


class ModelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.movies = load_catalog()
        cls.matrix = content_model(cls.movies)

    def test_content_excludes_seed_and_returns_five(self):
        title = self.movies.iloc[0].title
        result = content_recommendations(self.movies, self.matrix, title)
        self.assertEqual(len(result), 5)
        self.assertNotIn(title, result.title.tolist())
        self.assertEqual(result.movieId.nunique(), 5)

    def test_unknown_title(self):
        self.assertTrue(content_recommendations(self.movies, self.matrix, 'missing').empty)

    def test_collaborative_excludes_seen_movies(self):
        model = collaborative_model()
        users, movie_ids, matrix = model
        seen = set(movie_ids[matrix[0].indices])
        result = collaborative_recommendations(self.movies, model, users[0])
        self.assertEqual(len(result), 5)
        self.assertFalse(seen.intersection(result.movieId))
        self.assertTrue(collaborative_recommendations(self.movies, model, -1).empty)


class AppTestCase(unittest.TestCase):
    def setUp(self):
        # Deterministic API outage also verifies the local fallback paths.
        self.network = patch('requests.get', side_effect=__import__('requests').ConnectionError)
        self.network.start()
        self.groq_network = patch("requests.post", side_effect=__import__("requests").ConnectionError)
        self.groq_network.start()
        self.app = AppTest.from_file('app.py', default_timeout=30).run()

    def tearDown(self):
        self.network.stop()
        self.groq_network.stop()

    def test_home_filter_recommend_detail_and_back(self):
        app = self.app
        self.assertFalse(app.exception)
        app.selectbox[0].select('Sci-Fi').run()
        self.assertFalse(app.exception)
        app.button[0].click().run()
        self.assertEqual(len(app.session_state.recommendations), 5)
        app.button[1].click().run()
        self.assertIsNotNone(app.session_state.movie_page)
        self.assertFalse(app.exception)
        app.button[0].click().run()
        self.assertIsNone(app.session_state.movie_page)
        self.assertFalse(app.exception)

    def test_assistant_genre_and_movie_and_no_match(self):
        app = self.app
        for query in ['science fiction', 'Toy Story', 'zzzznoresults']:
            app.chat_input[0].set_value(query).run()
            self.assertFalse(app.exception)
            self.assertTrue(app.session_state.assistant_text_response)
        self.assertEqual(app.session_state.assistant_clickable_movies, [])

    def test_viewer_recommendations(self):
        app = self.app
        next(b for b in app.button if b.label == 'Find viewer recommendations').click().run()
        self.assertEqual(len(app.session_state['viewer_recommendations']), 5)
        self.assertFalse(app.exception)

    def test_ai_chat_history_and_clear(self):
        from unittest.mock import Mock
        import json
        response = Mock()
        response.json.return_value = {'choices': [{'message': {'content': json.dumps(
            {'reply': 'Arrival is thoughtful science fiction.', 'titles': ['Arrival']})}}]}
        with patch('movie_assistant.requests.post', return_value=response) as post:
            with patch.dict(os.environ, {'GROQ_API_KEY': 'test-key'}):
                app = AppTest.from_file('app.py', default_timeout=30).run()
                app.chat_input[0].set_value('Suggest science fiction').run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.session_state.conversation), 2)
                app.chat_input[0].set_value('Which is less scary?').run()
                self.assertEqual(len(app.session_state.conversation), 4)
                self.assertEqual(len(post.call_args.kwargs['json']['messages']), 4)
                next(b for b in app.button if b.label == 'Clear conversation').click().run()
                self.assertEqual(app.session_state.conversation, [])
                self.assertEqual(app.session_state.assistant_clickable_movies, [])

    def test_genre_word_boundaries(self):
        self.app.chat_input[0].set_value('information').run()
        self.assertEqual(self.app.session_state.assistant_clickable_movies, [])
        self.assertFalse(self.app.exception)


if __name__ == '__main__':
    unittest.main()
