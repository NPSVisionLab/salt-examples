var express = require('express');
var app = express();

app.use(express.static('app'));
app.use('/tiles', express.static('/home/trbatcha/salt/output'));

var server = app.listen(3000, function () {
  var port = server.address().port;
  console.log('Salt PNG example listening at http://localhost:%s', port);
});
