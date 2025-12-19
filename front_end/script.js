function getLocation() {
  if (!navigator.geolocation) {
    alert("Geolocation is not supported by your browser.");
    return;
  }

  navigator.geolocation.getCurrentPosition(
    (position) => {
      document.getElementById("latitude").value =
        position.coords.latitude.toFixed(6);
      document.getElementById("longitude").value =
        position.coords.longitude.toFixed(6);
    },
    () => {
      alert("Location access was denied or unavailable.");
    }
  );
}
