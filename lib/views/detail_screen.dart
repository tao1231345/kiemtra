import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../models/news_model.dart';
import '../providers/news_provider.dart';

class DetailScreen extends StatelessWidget {
  final News news;
  const DetailScreen({super.key, required this.news});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Chi tiết tin tức'),
        actions: [
          Consumer<NewsProvider>(
            builder: (context, provider, child) {
              return IconButton(
                icon: Icon(
                  news.isFavorite ? Icons.favorite : Icons.favorite_border,
                  color: news.isFavorite ? Colors.red : null,
                ),
                onPressed: () {
                  provider.toggleFavorite(news.id);
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(
                      content: Text(news.isFavorite ? 'Đã thêm vào yêu thích' : 'Đã xóa khỏi yêu thích'),
                      duration: const Duration(seconds: 1),
                    ),
                  );
                },
              );
            },
          )
        ],
      ),
      body: SingleChildScrollView(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Hero(
              tag: 'img-${news.id}',
              child: Image.network(news.imageUrl, width: double.infinity, height: 250, fit: BoxFit.cover),
            ),
            Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(news.title, style: const TextStyle(fontSize: 22, fontWeight: FontWeight.bold)),
                  const SizedBox(height: 10),
                  Row(
                    children: [
                      const Icon(Icons.calendar_today, size: 16, color: Colors.grey),
                      const SizedBox(width: 5),
                      Text(news.date, style: const TextStyle(color: Colors.grey)),
                    ],
                  ),
                  const Divider(height: 30),
                  Text(
                    news.body, 
                    style: const TextStyle(fontSize: 16, height: 1.5), 
                    textAlign: TextAlign.justify
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
