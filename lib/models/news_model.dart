class News {
  final String id;
  final String title;
  final String body;
  final String imageUrl;
  final String date;
  bool isFavorite;

  News({
    required this.id,
    required this.title,
    required this.body,
    required this.imageUrl,
    required this.date,
    this.isFavorite = false,
  });

  factory News.fromJson(Map<String, dynamic> json) {
    String img = '';
    
    if (json['thumbnail'] != null && json['thumbnail'].toString().isNotEmpty) {
      img = json['thumbnail'];
    }
    if (img.isEmpty && json['enclosure'] != null && json['enclosure']['link'] != null) {
      img = json['enclosure']['link'];
    }
    if (img.isEmpty) {
      final String desc = json['description'] ?? '';
      final RegExp regExp = RegExp(r'src="([^"]+)"');
      final match = regExp.firstMatch(desc);
      if (match != null) {
        img = match.group(1) ?? '';
      }
    }
    
    if (img.startsWith('//')) {
      img = 'https:$img';
    }

    if (img.isEmpty || img.contains('favicon') || !img.startsWith('http')) {
      img = 'https://picsum.photos/seed/${json['guid'].toString().hashCode}/400/300';
    }

    // Giải mã thực thể HTML (như &amp; thành &, &oacute; thành ó)
    String decodedTitle = _decodeHtmlEntities(json['title'] ?? 'Không có tiêu đề');
    String decodedBody = _decodeHtmlEntities(_stripHtml(json['description'] ?? 'Không có nội dung'));

    return News(
      id: json['guid'] ?? DateTime.now().toString(),
      title: decodedTitle,
      body: decodedBody,
      imageUrl: img,
      date: json['pubDate'] ?? '',
    );
  }

  static String _decodeHtmlEntities(String input) {
    // Bản đồ giải mã các thực thể HTML phổ biến trong tiếng Việt
    Map<String, String> entities = {
      '&amp;': '&',
      '&quot;': '"',
      '&apos;': "'",
      '&lt;': '<',
      '&gt;': '>',
      '&aacute;': 'á', '&agrave;': 'à', '&ả;': 'ả', '&ã;': 'ã', '&ạ;': 'ạ',
      '&acirc;': 'â', '&ấ;': 'ấ', '&ầ;': 'ầ', '&ẩ;': 'ẩ', '&ẫ;': 'ẫ', '&ậ;': 'ậ',
      '&ă;': 'ă', '&ắ;': 'ắ', '&ằ;': 'ằ', '&ẳ;': 'ẳ', '&ẵ;': 'ẵ', '&ặ;': 'ặ',
      '&eacute;': 'é', '&è;': 'è', '&ẻ;': 'ẻ', '&ẽ;': 'ẽ', '&ẹ;': 'ẹ',
      '&ecirc;': 'ê', '&ế;': 'ế', '&ề;': 'ề', '&ể;': 'ể', '&ễ;': 'ễ', '&ệ;': 'ệ',
      '&iacute;': 'í', '&ì;': 'ì', '&ỉ;': 'ỉ', '&ĩ;': 'ĩ', '&ị;': 'ị',
      '&oacute;': 'ó', '&ò;': 'ò', '&ỏ;': 'ỏ', '&õ;': 'õ', '&ọ;': 'ọ',
      '&ocirc;': 'ô', '&ố;': 'ố', '&ồ;': 'ồ', '&ổ;': 'ổ', '&ỗ;': 'ỗ', '&ộ;': 'ộ',
      '&ơ;': 'ơ', '&ớ;': 'ớ', '&ờ;': 'ờ', '&ở;': 'ở', '&ỡ;': 'ỡ', '&ợ;': 'ợ',
      '&uacute;': 'ú', '&ù;': 'ù', '&ủ;': 'ủ', '&ũ;': 'ũ', '&ụ;': 'ụ',
      '&ư;': 'ư', '&ứ;': 'ứ', '&ừ;': 'ừ', '&ử;': 'ử', '&ữ;': 'ữ', '&ự;': 'ự',
      '&ý;': 'ý', '&ỳ;': 'ỳ', '&ỷ;': 'ỷ', '&ỹ;': 'ỹ', '&ỵ;': 'ỵ',
      '&đ;': 'đ',
    };

    String output = input;
    // Xử lý các trường hợp đặc biệt bị mã hóa 2 lần (double encoding) như &amp;oacute;
    output = output.replaceAll('&amp;', '&');
    
    entities.forEach((entity, value) {
      output = output.replaceAll(entity, value);
    });
    
    return output;
  }

  static String _stripHtml(String htmlString) {
    return htmlString
        .replaceAll(RegExp(r'<img[^>]*>'), '')
        .replaceAll(RegExp(r'<[^>]*>|&nbsp;'), '')
        .trim();
  }
}
