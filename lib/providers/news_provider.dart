import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import '../models/news_model.dart';

class NewsProvider with ChangeNotifier {
  List<News> _newsList = [];
  List<News> _filteredNewsList = [];
  bool _isLoading = false;
  String _errorMessage = '';
  String _selectedCategory = 'Tất cả';

  List<News> get newsList => _filteredNewsList;
  bool get isLoading => _isLoading;
  String get errorMessage => _errorMessage;
  String get selectedCategory => _selectedCategory;

  List<News> get favoriteNews => _newsList.where((news) => news.isFavorite).toList();

  // Bản đồ các chuyên mục tin tức (Thanh Niên RSS)
  final Map<String, String> _categoryUrls = {
    'Tất cả': 'https://thanhnien.vn/rss/home.rss',
    'Công nghệ': 'https://thanhnien.vn/rss/cong-nghe-game.rss',
    'Kinh doanh': 'https://thanhnien.vn/rss/kinh-doanh.rss',
    'Thể thao': 'https://thanhnien.vn/rss/the-thao.rss',
    'Khoa học': 'https://thanhnien.vn/rss/doi-song/khoa-hoc.rss',
  };

  Future<void> fetchNews({String? category}) async {
    if (category != null) _selectedCategory = category;
    
    _isLoading = true;
    _errorMessage = '';
    notifyListeners();

    try {
      final url = _categoryUrls[_selectedCategory] ?? _categoryUrls['Tất cả']!;
      final apiUrl = 'https://api.rss2json.com/v1/api.json?rss_url=$url';
      
      final response = await http.get(Uri.parse(apiUrl)).timeout(const Duration(seconds: 20));

      if (response.statusCode == 200) {
        final Map<String, dynamic> data = json.decode(response.body);
        if (data['status'] == 'ok') {
          final List<dynamic> items = data['items'];
          _newsList = items.map((json) => News.fromJson(json)).toList();
          _filteredNewsList = _newsList;
        } else {
          _loadMockData();
        }
      } else {
        _loadMockData();
      }
    } catch (e) {
      _loadMockData();
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  void searchNews(String query) {
    if (query.isEmpty) {
      _filteredNewsList = _newsList;
    } else {
      _filteredNewsList = _newsList
          .where((news) => news.title.toLowerCase().contains(query.toLowerCase()))
          .toList();
    }
    notifyListeners();
  }

  void toggleFavorite(String id) {
    final index = _newsList.indexWhere((news) => news.id == id);
    if (index != -1) {
      _newsList[index].isFavorite = !_newsList[index].isFavorite;
      notifyListeners();
    }
  }

  void _loadMockData() {
    // Dữ liệu mẫu tiếng Việt theo chuyên mục để demo
    _newsList = [
      News(
        id: 'm1',
        title: 'Tin tức $_selectedCategory mới nhất hôm nay',
        body: 'Đây là nội dung tóm tắt cho chuyên mục $_selectedCategory được cập nhật liên tục...',
        imageUrl: 'https://picsum.photos/seed/$_selectedCategory/400/300',
        date: '16/04/2024',
      ),
      News(
        id: 'm2',
        title: 'Phát triển ứng dụng Flutter chuyên nghiệp',
        body: 'Hướng dẫn xây dựng kiến trúc Clean Architecture cho ứng dụng di động hiện đại...',
        imageUrl: 'https://picsum.photos/seed/flutter/400/300',
        date: '15/04/2024',
      ),
    ];
    _filteredNewsList = _newsList;
    _errorMessage = '';
  }
}
