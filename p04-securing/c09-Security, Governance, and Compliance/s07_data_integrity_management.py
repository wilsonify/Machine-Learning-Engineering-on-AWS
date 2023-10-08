"""
SELECT booking_changes, has_booking_changes, *
FROM dev.public.bookings
WHERE
(booking_changes=0 AND has_booking_changes='True')
OR
(booking_changes>0 AND has_booking_changes='False');

"""